// Same weights as jishui_200m.h with the production context shape.
#pragma once

#define MODEL_NAME "Jishui-200M"
#define DIM 704
#define HIDDEN 1856
#define HEADS 11
#define KV_HEADS 11
#define HD 64
#define GQA_RATIO (HEADS / KV_HEADS)  // 1:1 here, so Jishui-200M is MHA
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 2048
#define NLAYERS 30
#define VOCAB 32768
#define CKPT_PATH "ane_jishui_200m_ckpt.bin"
#define DEFAULT_DATA_PATH "dataset/processed"
#define RES_ALPHA 1.0f
#define USE_LAYER_NORM 1
#define ROPE_TRADITIONAL 0
#define DISABLE_VOCAB_COMPACTION 1
#define ADAM_EPS 1.0e-5f
