program skalaxc_fortran_assignment_test
   use skalaxc
   implicit none

   character(len=32) :: handle_type
   type(skalaxc_runtime_environment_t) :: rt_source, rt_destination
   type(skalaxc_molecule_t) :: mol_source, mol_destination
   type(skalaxc_basisset_t) :: basis_source, basis_destination
   type(skalaxc_molgrid_t) :: grid_source, grid_destination
   type(skalaxc_load_balancer_t) :: lb_source, lb_destination
   type(skalaxc_molecular_weights_t) :: weights_source, weights_destination
   type(skalaxc_functional_t) :: func_source, func_destination
   type(skalaxc_xc_integrator_t) :: xc_source, xc_destination

   if (command_argument_count() /= 1) error stop "expected one handle type"
   call get_command_argument(1, handle_type)

   select case (trim(handle_type))
   case ("runtime")
      rt_destination = rt_source
   case ("molecule")
      mol_destination = mol_source
   case ("basis")
      basis_destination = basis_source
   case ("grid")
      grid_destination = grid_source
   case ("load-balancer")
      lb_destination = lb_source
   case ("molecular-weights")
      weights_destination = weights_source
   case ("functional")
      func_destination = func_source
   case ("integrator")
      xc_destination = xc_source
   case default
      error stop "unknown handle type"
   end select

   error stop "copy assignment unexpectedly succeeded"
end program skalaxc_fortran_assignment_test
