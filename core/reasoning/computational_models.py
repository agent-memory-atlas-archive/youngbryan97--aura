"""Reviewed model cells with declared premises, units, and domains of validity."""

from core.reasoning.computational_knowledge import ComputationModel
from typing import Any

PHYSICS = "https://openstax.org/books/university-physics-volume-1"
THERMODYNAMICS = "https://openstax.org/books/university-physics-volume-2"
CHEMISTRY = "https://openstax.org/books/chemistry-2e"
ENZYME_KINETICS = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3381512/"


def _model(
    name: Any,
    field: Any,
    inputs: Any,
    unit: Any,
    expression: Any,
    assumptions: Any,
    reference: Any,
    nonnegative: tuple[Any, ...]=(),
    positive: tuple[Any, ...]=(),
) -> Any:
    return ComputationModel(name + ".v1", field, tuple(inputs), unit, expression,
                            tuple(assumptions), reference, tuple(nonnegative), tuple(positive))


MODEL_CELLS = {model.identity: model for model in (
    _model("arithmetic_sum", "mathematics", (("a", "count"), ("b", "count")), "count", "a+b", (), "rational arithmetic"),
    _model("arithmetic_difference", "mathematics", (("a", "count"), ("b", "count")), "count", "a-b", (), "rational arithmetic"),
    _model("rectangular_area", "geometry", (("width", "m"), ("height", "m")), "m^2", "width*height",
           ("planar_rectangle",), "Euclidean rectangle area", ("width", "height")),
    _model("squared_distance_2d", "geometry", (("dx", "m"), ("dy", "m")), "m^2", "dx**2+dy**2",
           ("euclidean_coordinates",), "Euclidean squared distance"),
    _model("average_speed_distance", "motion", (("v", "m/s"), ("t", "s")), "m", "v*t",
           ("average_speed_over_interval",), PHYSICS + "/pages/3-2-instantaneous-velocity-and-speed", ("v", "t")),
    _model("constant_acceleration_position", "motion", (("x0", "m"), ("v0", "m/s"), ("a", "m/s^2"), ("t", "s")),
           "m", "x0+v0*t+a*t**2/2", ("constant_acceleration", "inertial_frame"),
           PHYSICS + "/pages/3-4-motion-with-constant-acceleration", ("t",)),
    _model("newton_force", "mechanics", (("m", "kg"), ("a", "m/s^2")), "N", "m*a",
           ("classical_constant_mass", "inertial_frame"), PHYSICS + "/pages/5-3-newtons-second-law", positive=("m",)),
    _model("kinetic_energy", "mechanics", (("m", "kg"), ("v", "m/s")), "J", "m*v**2/2",
           ("nonrelativistic_speed",), PHYSICS + "/pages/7-2-kinetic-energy", ("m",)),
    _model("hydrostatic_pressure", "water", (("rho", "kg/m^3"), ("g", "m/s^2"), ("h", "m")), "Pa", "rho*g*h",
           ("static_uniform_density_fluid", "uniform_gravity"), PHYSICS + "/pages/14-1-fluids-density-and-pressure",
           ("g", "h"), ("rho",)),
    _model("buoyancy", "water", (("rho", "kg/m^3"), ("g", "m/s^2"), ("volume", "m^3")), "N", "rho*g*volume",
           ("static_uniform_density_fluid", "uniform_gravity"), PHYSICS + "/pages/14-4-archimedes-principle-and-buoyancy",
           ("g", "volume"), ("rho",)),
    _model("continuity_speed", "fluids", (("flow", "m^3/s"), ("area", "m^2")), "m/s", "flow/area",
           ("steady_incompressible_flow",), PHYSICS + "/pages/14-5-fluid-dynamics", positive=("area",)),
    _model("reynolds_number", "fluids", (("rho", "kg/m^3"), ("speed", "m/s"), ("diameter", "m"), ("mu", "Pa s")),
           "count", "rho*speed*diameter/mu", ("newtonian_fluid",), PHYSICS + "/pages/14-7-viscosity-and-turbulence",
           ("speed",), ("rho", "diameter", "mu")),
    _model("sensible_heat", "thermal", (("m", "kg"), ("c", "J/(kg K)"), ("dT", "K")), "J", "m*c*dT",
           ("constant_specific_heat", "no_phase_transition"), THERMODYNAMICS + "/pages/1-4-heat-transfer-specific-heat-and-calorimetry",
           ("m",), ("c",)),
    _model("ohmic_current", "electrical", (("voltage", "V"), ("resistance", "ohm")), "A", "voltage/resistance",
           ("ohmic_element_at_fixed_conditions",), THERMODYNAMICS + "/pages/9-4-ohms-law", positive=("resistance",)),
    _model("electrical_power", "electrical", (("voltage", "V"), ("current", "A")), "W", "voltage*current",
           ("instantaneous_values_same_element",), THERMODYNAMICS + "/pages/9-5-electrical-energy-and-power"),
    _model("pressure_force", "mechanics", (("pressure", "Pa"), ("area", "m^2")), "N", "pressure*area",
           ("uniform_normal_pressure",), PHYSICS + "/pages/14-1-fluids-density-and-pressure", ("area",)),
    _model("mechanical_work", "mechanics", (("force", "N"), ("distance", "m")), "J", "force*distance",
           ("constant_force_parallel_to_displacement",), PHYSICS + "/pages/7-1-work", ("distance",)),
    _model("hydraulic_power", "fluids", (("pressure", "Pa"), ("flow", "m^3/s")), "W", "pressure*flow",
           ("steady_incompressible_flow",), PHYSICS + "/pages/14-6-bernoullis-equation"),
    _model("density_mass", "materials", (("rho", "kg/m^3"), ("volume", "m^3")), "kg", "rho*volume",
           ("uniform_density",), PHYSICS + "/pages/14-1-fluids-density-and-pressure", ("volume",), ("rho",)),
    _model("ideal_gas_pressure", "chemistry", (("n", "mol"), ("R", "J/(mol K)"), ("T", "K"), ("volume", "m^3")),
           "Pa", "n*R*T/volume", ("ideal_gas_regime", "equilibrium_temperature"),
           THERMODYNAMICS + "/pages/2-1-molecular-model-of-an-ideal-gas", ("n",), ("R", "T", "volume")),
    _model("molar_concentration", "chemistry", (("n", "mol"), ("volume", "m^3")), "mol/m^3", "n/volume",
           ("homogeneous_solution",), CHEMISTRY + "/pages/3-3-molarity", ("n",), ("volume",)),
    _model("dilution_concentration", "chemistry", (("c1", "mol/m^3"), ("v1", "m^3"), ("v2", "m^3")),
           "mol/m^3", "c1*v1/v2", ("solute_conserved", "homogeneous_solution"),
           CHEMISTRY + "/pages/3-3-molarity", ("c1", "v1"), ("v2",)),
    _model("michaelis_menten_rate", "biology", (("vmax", "mol/(m^3 s)"), ("substrate", "mol/m^3"), ("km", "mol/m^3")),
           "mol/(m^3 s)", "vmax*substrate/(km+substrate)",
           ("single_substrate_noncooperative_kinetics", "initial_rate_steady_state"), ENZYME_KINETICS,
           ("vmax", "substrate"), ("km",)),
)}
