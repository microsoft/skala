Error handling in GauXC
=======================

This section provides a reference for error handling in GauXC, including C++ exceptions, C status handles, and Fortran status handles.

C++ exceptions
--------------

.. cpp:class:: GauXC::generic_gauxc_exception : public std::exception

   Base class for exceptions thrown by GauXC. This class inherits from std::exception and can be used to catch all exceptions thrown by GauXC.

   .. cpp:function:: const char* what() const noexcept override

      Get a human-readable error message describing the exception.

      :returns: A pointer to a null-terminated string containing the error message.


C status handle
---------------

.. c:struct:: GauXCStatus

   .. c:member:: int code

      The error code of the status. A value of 0 indicates success, while non-zero values indicate different types of errors.

   .. c:member:: char* message

      A human-readable error message providing more details about the error.
      This is a null-terminated string that should be freed by the caller when no longer needed.


Fortran status handle
---------------------

.. f:module:: gauxc_status
   :synopsis: Fortran bindings for GauXC status handling

.. f:currentmodule:: gauxc_status

.. f:type:: gauxc_status_type

   Representation of a status handle in the GauXC Fortran API, containing an error code and a message.

   :f integer(c_int) code: The error code of the status. A value of 0 indicates success, while non-zero values indicate different types of errors.
   :f type(c_ptr) message: A pointer to a null-terminated string containing a human-readable error message providing more details about the error. This string should be freed by the caller when no longer needed.

   .. f:function:: gauxc_status_message(status)

      Get the error message from a GauXCStatus variable.

      :param type(gauxc_status_type) status: The GauXCStatus variable from which to retrieve the error message.
      :returns character(len=*): Error message string.