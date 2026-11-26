# Helper utilities to fetch .fun models used by SkalaXC tests/examples.
#
# Hashes below are sourced from Hugging Face model metadata (LFS sha256):
#   https://huggingface.co/api/models/<repo>?blobs=true

function(skalaxc_download_baseline_models model_dir)
  if(NOT model_dir)
    message(FATAL_ERROR "skalaxc_download_baseline_models requires a model directory")
  endif()

  if(ARGC LESS 2)
    message(FATAL_ERROR
      "skalaxc_download_baseline_models requires at least one model filename")
  endif()

  file(MAKE_DIRECTORY "${model_dir}")

  foreach(_model_name IN LISTS ARGN)
    set(_model_path "${model_dir}/${_model_name}")
    set(_remote_model_name "${_model_name}")

    if(_model_name STREQUAL "ldax.fun")
      set(_repo_id "microsoft/skala-baselines")
      set(_expected_sha256 "dd30928579ac970ffccc0c6f4ff6e2f7d7eeda3665352229a3f2678b3d0cb32e")
    elseif(_model_name STREQUAL "pbe.fun")
      set(_repo_id "microsoft/skala-baselines")
      set(_expected_sha256 "da4da7dfed02bde938606c77b38e785283f15ad3be69159051e11cf213a97862")
    elseif(_model_name STREQUAL "tpss.fun")
      set(_repo_id "microsoft/skala-baselines")
      set(_expected_sha256 "c2775e8c9512e399e8b35f1d4424052fdafaa8631a8e638a28ff460619999449")
    elseif(_model_name STREQUAL "skala-1.1.fun")
      set(_repo_id "microsoft/skala-1.1")
      set(_remote_model_name "skala-1.1-rev1.fun")
      set(_expected_sha256 "7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd")
    elseif(_model_name STREQUAL "skala-1.1-cuda.fun")
      set(_repo_id "microsoft/skala-1.1")
      set(_remote_model_name "skala-1.1-rev1-cuda.fun")
      set(_expected_sha256 "f848eae769dca91741a518ae7275d10caac398ab21db649f91bc1f136872f223")
    else()
      message(FATAL_ERROR
        "No trusted hash configured for model ${_model_name}. "
        "Refusing to download without verification.")
    endif()

    if(EXISTS "${_model_path}")
      file(SHA256 "${_model_path}" _actual_sha256)
      if(NOT _actual_sha256 STREQUAL _expected_sha256)
        message(FATAL_ERROR
          "Hash verification failed for existing model ${_model_path}.\n"
          "Expected: ${_expected_sha256}\n"
          "Actual:   ${_actual_sha256}\n"
          "Remove the file and reconfigure to re-download a trusted copy.")
      endif()
      continue()
    endif()

    set(_url "https://huggingface.co/${_repo_id}/resolve/main/${_remote_model_name}")
    message(STATUS "SkalaXC: downloading ${_model_name}")

    file(
      DOWNLOAD "${_url}" "${_model_path}"
      SHOW_PROGRESS
      EXPECTED_HASH "SHA256=${_expected_sha256}"
      STATUS _download_status
      LOG _download_log
      TLS_VERIFY ON
    )

    list(GET _download_status 0 _download_code)
    list(GET _download_status 1 _download_message)
    if(NOT _download_code EQUAL 0)
      message(FATAL_ERROR
        "Failed to download ${_model_name} from ${_url}: ${_download_message}\n"
        "CMake download log:\n${_download_log}")
    endif()
  endforeach()

endfunction()
