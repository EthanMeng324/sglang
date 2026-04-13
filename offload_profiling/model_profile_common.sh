#!/usr/bin/env bash

_first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -d "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

profile_model_is_image() {
    case "${PROFILE_MODEL:-}" in
        flux|flux_2)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_profile_model() {
    local mode="$1"
    local model_arg="${2:-}"

    PROFILE_MODEL="${model_arg:-${PROFILE_MODEL:-wanvideo}}"
    PROFILE_MODEL="$(printf '%s' "$PROFILE_MODEL" | tr '[:upper:]' '[:lower:]')"

    local default_model_path=""
    local model_path_hint="MODEL_PATH"
    case "$PROFILE_MODEL" in
        wan|wanvideo)
            PROFILE_MODEL="wanvideo"
            default_model_path="${WANVIDEO_MODEL_PATH:-$(_first_existing_dir \
                /scratch/user/u.hm347392/model/Wan2.2-T2V-A14B-Diffusers \
                /scratch/user/u.hm347392/models/Wan2.2-T2V-A14B-Diffusers \
            || true)}"
            model_path_hint="MODEL_PATH or WANVIDEO_MODEL_PATH"
            ;;
        flux|flux1|flux_1|fluximage|flux_image)
            PROFILE_MODEL="flux"
            default_model_path="${FLUX_MODEL_PATH:-$(_first_existing_dir \
                /scratch/user/u.hm347392/model/FLUX.1-dev \
                /scratch/user/u.hm347392/models/FLUX.1-dev \
            || true)}"
            model_path_hint="MODEL_PATH or FLUX_MODEL_PATH"
            ;;
        flux_2|flux2|flux-2|flux2dev|flux_2_dev)
            PROFILE_MODEL="flux_2"
            default_model_path="${FLUX_2_MODEL_PATH:-$(_first_existing_dir \
                /scratch/user/u.hm347392/model/FLUX.2-klein-4B \
                /scratch/user/u.hm347392/models/FLUX.2-klein-4B \
            || true)}"
            model_path_hint="MODEL_PATH or FLUX_2_MODEL_PATH"
            ;;
        hunyuan|hunyuanvideo)
            PROFILE_MODEL="hunyuanvideo"
            default_model_path="${HUNYUANVIDEO_MODEL_PATH:-$(_first_existing_dir \
                /scratch/user/u.hm347392/model/HunyuanVideo \
                /scratch/user/u.hm347392/models/HunyuanVideo \
            || true)}"
            model_path_hint="MODEL_PATH or HUNYUANVIDEO_MODEL_PATH"
            ;;
        *)
            echo "ERROR: unsupported PROFILE_MODEL='$PROFILE_MODEL'. Supported: wanvideo, flux, flux_2, hunyuanvideo."
            return 1
            ;;
    esac

    MODEL_PATH="${MODEL_PATH:-$default_model_path}"
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "ERROR: no model path configured for PROFILE_MODEL='$PROFILE_MODEL'. Set ${model_path_hint}."
        return 1
    fi

    PROMPT="${PROMPT:-A cat walks on the grass, realistic}"

    local mode_stem
    case "$mode" in
        new) mode_stem="new_offload" ;;
        no) mode_stem="no_offload" ;;
        old) mode_stem="old_offload" ;;
        ratio) mode_stem="ratio_resident_offload" ;;
        phase) mode_stem="ratio_resident_offload" ;;
        *)
            echo "ERROR: unsupported profiling mode '$mode'."
            return 1
            ;;
    esac

    MEDIA_EXT="mp4"
    if profile_model_is_image; then
        MEDIA_EXT="png"
    fi

    NSYS_BASENAME="${mode_stem}_nsys"
    ARTIFACT_BASENAME="nsys_${mode_stem}_profiled"
    PERF_BASENAME="perf_${mode_stem}_profiled.json"
    if [[ "$PROFILE_MODEL" != "wanvideo" ]]; then
        NSYS_BASENAME="${PROFILE_MODEL}_${mode_stem}_nsys"
        ARTIFACT_BASENAME="nsys_${PROFILE_MODEL}_${mode_stem}_profiled"
        PERF_BASENAME="perf_${PROFILE_MODEL}_${mode_stem}_profiled.json"
    fi
}

apply_profile_model_defaults() {
    PROMPT="${PROMPT:-A cat walks on the grass, realistic}"
    NUM_GPUS="${NUM_GPUS:-2}"
    ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage_attn}"
    NUM_OUTPUTS_PER_PROMPT="${NUM_OUTPUTS_PER_PROMPT:-${BATCH_SIZE:-1}}"
    if ! [[ "$NUM_OUTPUTS_PER_PROMPT" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: NUM_OUTPUTS_PER_PROMPT/BATCH_SIZE must be a positive integer."
        return 1
    fi
    BATCH_SIZE="${BATCH_SIZE:-$NUM_OUTPUTS_PER_PROMPT}"

    case "$PROFILE_MODEL" in
        flux|flux_2)
            NUM_FRAMES="${NUM_FRAMES:-1}"
            HEIGHT="${HEIGHT:-1024}"
            WIDTH="${WIDTH:-1024}"
            NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
            GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
            GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-}"
            ;;
        wanvideo)
            NUM_FRAMES="${NUM_FRAMES:-81}"
            HEIGHT="${HEIGHT:-720}"
            WIDTH="${WIDTH:-1280}"
            NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
            GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
            GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-3.0}"
            ;;
        hunyuanvideo)
            NUM_FRAMES="${NUM_FRAMES:-81}"
            HEIGHT="${HEIGHT:-720}"
            WIDTH="${WIDTH:-1280}"
            NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
            GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
            GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-}"
            ;;
    esac
}

apply_profile_output_layout() {
    MODEL_RESULTS_DIR="${RESULTS_DIR%/}/${PROFILE_MODEL}"
    MODEL_PROFILES_DIR="${PROFILES_DIR%/}/${PROFILE_MODEL}"
    mkdir -p "$MODEL_RESULTS_DIR" "$MODEL_PROFILES_DIR"

    NSYS_OUTPUT_PREFIX="${NSYS_OUTPUT_PREFIX:-$MODEL_PROFILES_DIR/$NSYS_BASENAME}"
    local default_output_file_name="${ARTIFACT_BASENAME}.${MEDIA_EXT}"
    if [[ -n "${VIDEO_OUT:-}" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$VIDEO_OUT")}"
        OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME:-$(basename "$VIDEO_OUT")}"
    else
        OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_RESULTS_DIR}"
        OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME:-$default_output_file_name}"
    fi
    mkdir -p "$OUTPUT_DIR"
    VIDEO_OUT="${OUTPUT_DIR%/}/${OUTPUT_FILE_NAME}"
    PERF_OUT="${PERF_OUT:-$MODEL_PROFILES_DIR/$PERF_BASENAME}"
}

format_output_display_path() {
    local output_path="$1"
    local num_outputs="${2:-1}"

    if [[ "$num_outputs" -le 1 ]]; then
        printf '%s' "$output_path"
        return 0
    fi

    local output_dir
    local output_file
    output_dir="$(dirname "$output_path")"
    output_file="$(basename "$output_path")"

    if [[ "$output_file" == *.* ]]; then
        printf '%s/%s_<idx>.%s' "$output_dir" "${output_file%.*}" "${output_file##*.}"
    else
        printf '%s/%s_<idx>' "$output_dir" "$output_file"
    fi
}
