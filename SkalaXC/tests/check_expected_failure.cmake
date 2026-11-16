if(NOT DEFINED EXECUTABLE OR NOT DEFINED HANDLE_TYPE OR
   NOT DEFINED EXPECTED_MESSAGE)
  message(FATAL_ERROR
    "EXECUTABLE, HANDLE_TYPE, and EXPECTED_MESSAGE are required")
endif()

execute_process(
  COMMAND "${EXECUTABLE}" "${HANDLE_TYPE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE standard_output
  ERROR_VARIABLE standard_error)

if(result EQUAL 0)
  message(FATAL_ERROR
    "${HANDLE_TYPE} assignment unexpectedly succeeded")
endif()

set(output "${standard_output}\n${standard_error}")
string(FIND "${output}" "${EXPECTED_MESSAGE}" message_position)
if(message_position EQUAL -1)
  message(FATAL_ERROR
    "${HANDLE_TYPE} assignment failed without the expected diagnostic:\n${output}")
endif()
