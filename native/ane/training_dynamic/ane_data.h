// On-demand train sampler for the frozen Jishui NPY shards.
//
// The Python exporter writes this file format; keeping the loader here avoids
// copying billions of uint16 ids into a second monolithic stream.  It mirrors
// PackedBatchIterator's token-weighted category sampling and EOD packing.
#pragma once

#include "config.h"
#include <stdint.h>
#include <limits.h>
#include <ctype.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The native ANE loader requires a little-endian host"
#endif

typedef struct {
    void *mapping;
    size_t mapping_len;
    const uint16_t *tokens;
    size_t n_tokens;
} AneNpyShard;

typedef struct {
    uint32_t shard;
    uint64_t offset;
    uint32_t length;
    uint8_t category;
} AneDoc;

typedef struct {
    AneDoc *docs;
    size_t n_docs;
    AneNpyShard shards[64];
    int n_shards;
    uint32_t *group_docs[7];
    uint64_t *group_cum[7];
    size_t group_n[7];
    uint64_t group_total[7];
    double group_prob[7];
    uint32_t eod_token_id;
    uint16_t *sample;
} AneTrainIndex;

// Stateless-per-step sampling RNG.  Seeding from the absolute microstep
// makes a resumed native run choose the same next window without putting an
// opaque libc ``drand48`` state in the checkpoint format.
typedef struct { uint64_t state; } AneRng;

static inline void ane_rng_seed(AneRng *rng, uint64_t seed) {
    // splitmix64 initialization, with a nonzero state for xorshift below.
    uint64_t z = seed + UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    rng->state = (z ^ (z >> 31)) | 1;
}

static inline uint64_t ane_rng_next(AneRng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static inline double ane_rng_uniform(AneRng *rng) {
    return (double)(ane_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t category_count;
    uint32_t reserved;
    uint64_t records;
} AneIndexHeader;

// Version 2 adds the target category mixture and EOD id used by the native
// sampler.  Version 1 remains readable for the first frozen index export.
typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t category_count;
    uint32_t reserved;
    uint64_t records;
    double target_mix[6];
    uint32_t eod_token_id;
    uint32_t reserved2;
} AneIndexHeaderV2;

// The on-disk record is Python's ``struct.Struct("<IQIB3x")``.  Do not
// rely on the host ABI here: a naturally aligned C struct is 24 bytes on
// arm64, while the frozen index is 20 bytes per record.
typedef struct __attribute__((packed)) {
    uint32_t shard;
    uint64_t offset;
    uint32_t length;
    uint8_t category;
    uint8_t reserved[3];
} AneIndexRecord;

_Static_assert(sizeof(AneIndexHeader) == 32, "ANE index header layout changed");
_Static_assert(sizeof(AneIndexHeaderV2) == 88, "ANE v2 index header layout changed");
_Static_assert(sizeof(AneIndexRecord) == 20, "ANE index record must be 20 bytes");

static bool ane_read_npy_shard(AneNpyShard *out, const char *data_dir, int shard_id) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/tokens/shard_%05d.npy", data_dir, shard_id);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) { close(fd); return false; }
    unsigned char pre[16] = {0};
    if (pread(fd, pre, sizeof(pre), 0) < 10 || pre[0] != 0x93 ||
        memcmp(pre + 1, "NUMPY", 5) != 0) {
        close(fd); printf("not an NPY shard: %s\n", path); return false;
    }
    size_t header_bytes;
    size_t header_len;
    if (pre[6] == 1) {
        header_len = (size_t)pre[8] | ((size_t)pre[9] << 8);
        header_bytes = 10;
    } else if (pre[6] == 2 || pre[6] == 3) {
        if (pread(fd, pre, 16, 0) < 12) { close(fd); return false; }
        header_len = (size_t)pre[8] | ((size_t)pre[9] << 8) |
                     ((size_t)pre[10] << 16) | ((size_t)pre[11] << 24);
        header_bytes = 12;
    } else {
        close(fd); printf("unsupported NPY version in %s\n", path); return false;
    }
    if (header_len > (size_t)st.st_size - header_bytes || header_len > (1u << 20)) {
        close(fd); printf("invalid NPY header length in %s\n", path); return false;
    }
    size_t data_offset = header_bytes + header_len;
    if (data_offset >= (size_t)st.st_size || ((st.st_size - data_offset) & 1)) {
        close(fd); printf("invalid NPY payload in %s\n", path); return false;
    }
    // The pipeline freezes little-endian one-dimensional uint16 NPY shards.
    // Check the textual header before interpreting the mapped bytes; without
    // this guard a stale float32 or Fortran-order shard would look plausible.
    char *header = (char *)malloc(header_len + 1);
    if (!header || pread(fd, header, header_len, (off_t)header_bytes) != (ssize_t)header_len) {
        free(header); close(fd); return false;
    }
    header[header_len] = '\0';
    bool dtype_ok = strstr(header, "'descr': '<u2'") || strstr(header, "\"descr\": '<u2'") ||
                    strstr(header, "'descr': \"<u2\"") || strstr(header, "\"descr\": \"<u2\"");
    bool order_ok = strstr(header, "'fortran_order': False") || strstr(header, "\"fortran_order\": False");
    char *shape_key = strstr(header, "'shape'");
    if (!shape_key) shape_key = strstr(header, "\"shape\"");
    char *shape_open = shape_key ? strchr(shape_key, '(') : NULL;
    char *shape_end = NULL;
    errno = 0;
    unsigned long long declared_tokens = shape_open ? strtoull(shape_open + 1, &shape_end, 10) : 0;
    bool shape_ok = shape_open && shape_end && shape_end != shape_open + 1 && errno == 0;
    if (shape_ok) {
        while (*shape_end && isspace((unsigned char)*shape_end)) shape_end++;
        if (*shape_end == ',') shape_end++;
        while (*shape_end && isspace((unsigned char)*shape_end)) shape_end++;
        shape_ok = (*shape_end == ')') && declared_tokens == ((size_t)st.st_size - data_offset) / sizeof(uint16_t);
    }
    free(header);
    if (!dtype_ok || !order_ok || !shape_ok) {
        close(fd); printf("unsupported NPY dtype/order in %s\n", path); return false;
    }
    void *mapping = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) { perror("mmap NPY"); return false; }
    out->mapping = mapping;
    out->mapping_len = (size_t)st.st_size;
    out->tokens = (const uint16_t *)((const uint8_t *)mapping + data_offset);
    out->n_tokens = ((size_t)st.st_size - data_offset) / sizeof(uint16_t);
    return true;
}

static bool ane_index_load(AneTrainIndex *index, const char *index_path,
                           const char *data_dir) {
    memset(index, 0, sizeof(*index));
    FILE *f = fopen(index_path, "rb");
    if (!f) { perror(index_path); return false; }
    AneIndexHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, "JSHANEI1", 8) != 0 ||
        (h.version != 1 && h.version != 2) || h.vocab_size != VOCAB ||
        h.category_count != 6 || h.records == 0 ||
        h.records > SIZE_MAX / sizeof(AneDoc) || h.records > UINT32_MAX) {
        fclose(f); printf("invalid ANE index header: %s\n", index_path); return false;
    }
    index->eod_token_id = 2;
    double target[7] = {0.0, .35, .15, .08, .12, .15, .15};
    if (h.version == 2) {
        AneIndexHeaderV2 v2;
        memset(&v2, 0, sizeof(v2));
        memcpy(v2.magic, h.magic, sizeof(h.magic));
        v2.version = h.version; v2.vocab_size = h.vocab_size;
        v2.category_count = h.category_count; v2.reserved = h.reserved;
        v2.records = h.records;
        if (fread(v2.target_mix, sizeof(v2.target_mix), 1, f) != 1 ||
            fread(&v2.eod_token_id, sizeof(v2.eod_token_id), 1, f) != 1 ||
            fread(&v2.reserved2, sizeof(v2.reserved2), 1, f) != 1 ||
            v2.eod_token_id >= VOCAB) {
            fclose(f); printf("invalid ANE v2 index metadata: %s\n", index_path); return false;
        }
        for (int c = 1; c <= 6; c++) {
            if (!isfinite(v2.target_mix[c - 1]) || v2.target_mix[c - 1] < 0.0) {
                fclose(f); printf("invalid target mix in ANE index: %s\n", index_path); return false;
            }
            target[c] = v2.target_mix[c - 1];
        }
        index->eod_token_id = v2.eod_token_id;
    }
    struct stat index_stat;
    if (fstat(fileno(f), &index_stat) != 0) {
        fclose(f); return false;
    }
    uint64_t header_size = h.version == 2 ? sizeof(AneIndexHeaderV2) : sizeof(AneIndexHeader);
    uint64_t expected_size = header_size + h.records * (uint64_t)sizeof(AneIndexRecord);
    if ((uint64_t)index_stat.st_size != expected_size) {
        fclose(f); printf("unexpected ANE index size: got %llu expected %llu\n",
                          (unsigned long long)index_stat.st_size,
                          (unsigned long long)expected_size);
        return false;
    }
    index->n_docs = (size_t)h.records;
    index->docs = (AneDoc *)calloc(index->n_docs, sizeof(AneDoc));
    if (!index->docs) { fclose(f); return false; }
    size_t counts[7] = {0};
    int max_shard = -1;
    for (size_t i = 0; i < index->n_docs; i++) {
        AneIndexRecord r;
        if (fread(&r, sizeof(r), 1, f) != 1 || r.shard >= 64 ||
            r.category < 1 || r.category > 6 || r.length == 0) {
            fclose(f); return false;
        }
        index->docs[i].shard = r.shard;
        index->docs[i].offset = r.offset;
        index->docs[i].length = r.length;
        index->docs[i].category = r.category;
        counts[r.category]++;
        if ((int)r.shard > max_shard) max_shard = (int)r.shard;
    }
    fclose(f);
    if (max_shard >= (int)(sizeof(index->shards) / sizeof(index->shards[0]))) {
        printf("too many NPY shards\n"); return false;
    }
    index->n_shards = max_shard + 1;
    for (int s = 0; s <= max_shard; s++) {
        if (!ane_read_npy_shard(&index->shards[s], data_dir, s)) return false;
    }
    // Validate every document before making it eligible for sampling.  This
    // catches stale indexes and prevents an invalid token id from reaching
    // the embedding lookup (which would otherwise be an out-of-bounds read).
    for (size_t i = 0; i < index->n_docs; i++) {
        AneDoc *doc = &index->docs[i];
        if (doc->shard >= (uint32_t)index->n_shards ||
            doc->offset > index->shards[doc->shard].n_tokens ||
            (uint64_t)doc->length > index->shards[doc->shard].n_tokens - doc->offset) {
            printf("invalid document range in ANE index at record %zu\n", i);
            return false;
        }
    }
    for (int c = 1; c <= 6; c++) {
        index->group_n[c] = counts[c];
        if (!counts[c]) continue;
        if (counts[c] > SIZE_MAX / sizeof(uint32_t) || counts[c] > SIZE_MAX / sizeof(uint64_t)) {
            return false;
        }
        index->group_docs[c] = (uint32_t *)malloc(counts[c] * sizeof(uint32_t));
        index->group_cum[c] = (uint64_t *)malloc(counts[c] * sizeof(uint64_t));
        if (!index->group_docs[c] || !index->group_cum[c]) {
            return false;
        }
        size_t at = 0;
        uint64_t total = 0;
        for (size_t i = 0; i < index->n_docs; i++) {
            if (index->docs[i].category != c) continue;
            index->group_docs[c][at] = (uint32_t)i;
            if (UINT64_MAX - total < index->docs[i].length) {
                return false;
            }
            total += index->docs[i].length;
            index->group_cum[c][at++] = total;
        }
        index->group_total[c] = total;
    }
    // Missing categories are removed and the remaining probabilities are
    // renormalized exactly as PackedBatchIterator does.
    double sum = 0.0;
    for (int c = 1; c <= 6; c++) if (index->group_n[c]) sum += target[c];
    if (!isfinite(sum) || sum <= 0.0) return false;
    for (int c = 1; c <= 6; c++)
        if (index->group_n[c] && index->group_total[c] == 0) return false;
    for (int c = 1; c <= 6; c++) index->group_prob[c] =
        index->group_n[c] ? target[c] / sum : 0.0;
    if (index->eod_token_id >= VOCAB || (size_t)(SEQ + 1) > SIZE_MAX / sizeof(uint16_t)) return false;
    index->sample = (uint16_t *)malloc((size_t)(SEQ + 1) * sizeof(uint16_t));
    if (!index->sample) return false;
    return true;
}

static bool ane_index_sample(AneTrainIndex *index, AneRng *rng) {
    int category = 0;
    double r = ane_rng_uniform(rng);
    for (int c = 1; c <= 6; c++) {
        if (index->group_prob[c] <= 0.0) continue;
        r -= index->group_prob[c];
        if (r <= 0.0) { category = c; break; }
    }
    if (!category) for (int c = 1; c <= 6; c++) if (index->group_n[c]) { category = c; break; }
    size_t position = 0;
    while (position < (size_t)(SEQ + 1)) {
        uint64_t total = index->group_total[category];
        uint64_t token_position = (uint64_t)(ane_rng_uniform(rng) * (double)total);
        if (token_position >= total) token_position = total - 1;
        size_t lo = 0, hi = index->group_n[category];
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (index->group_cum[category][mid] <= token_position) lo = mid + 1;
            else hi = mid;
        }
        size_t local = lo;
        uint32_t doc_id = index->group_docs[category][local];
        AneDoc *doc = &index->docs[doc_id];
        uint64_t previous = local ? index->group_cum[category][local - 1] : 0;
        uint32_t start = (uint32_t)(token_position - previous);
        uint32_t available = doc->length - start;
        size_t count = available;
        if (count > (size_t)(SEQ + 1) - position) count = (size_t)(SEQ + 1) - position;
        if (doc->shard >= (uint32_t)index->n_shards ||
            (uint64_t)doc->offset + start + count > index->shards[doc->shard].n_tokens) return false;
        memcpy(index->sample + position,
               index->shards[doc->shard].tokens + doc->offset + start,
               count * sizeof(uint16_t));
        for (size_t i = 0; i < count; i++) {
            if (index->sample[position + i] >= VOCAB) {
                printf("token id %u exceeds vocab %d\n",
                       (unsigned)index->sample[position + i], VOCAB);
                return false;
            }
        }
        position += count;
        if (count == available && position < (size_t)(SEQ + 1))
            index->sample[position++] = (uint16_t)index->eod_token_id;
    }
    return true;
}

static void ane_index_free(AneTrainIndex *index) {
    if (!index) return;
    for (int s = 0; s < index->n_shards; s++)
        if (index->shards[s].mapping) munmap(index->shards[s].mapping, index->shards[s].mapping_len);
    for (int c = 1; c <= 6; c++) { free(index->group_docs[c]); free(index->group_cum[c]); }
    free(index->sample); free(index->docs);
    memset(index, 0, sizeof(*index));
}
