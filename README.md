# Wildfire model
This is a general model for wildfire propagation based on Finsler geometry. Contents:

## general_model.pdf
Open arXiv preprint of the following research paper: M. Á. Javaloyes, E. Pendás-Recondo and M. Sánchez. A General Model for Wildfire Propagation with Wind and Slope. SIAM J. Appl. Algebra Geom. 7(2), 414-439 (2023). All the codes are based on this model. Specifically, the velocity of the fire at each point, time and direction is given by the indicatrix of the Finsler metric F in Eq. (13), which depends on the parameters a, h, epsilon, phi and z(x,y). The main result Theorem 3.1 provides the ODE system Eq. (18), from which one can compute the fire trajectories, given F. The codes detailed below solve this ODE system and depict the evolution of the wildfire in several examples.

fig5_1.nb, fig5_2.nb, fig6.nb, fig7.nb, fig8.nb and fig9.nb are codes for Mathematica that provide the corresponding figures in general_model.pdf. The ODE system is solved using the default Mathematica ODE solver. The code is the same in all files, except for the values of the parameters of the model, which are specified for each case in the associated figure caption in general_model.pdf. The inputs (parameters) need to be smooth functions.

## model.py
Code for Python that provides the wildfire spread given discretized inputs, which are then linearly interpolated when necessary. No physical model is assumed here, so every quantity is dimensionless. The ODE system is solved using the Runge-Kutta 4 method. Inputs:
- Dimensions (t,x,y)
	- ti,tf,nt: floats. Initial time, final time and total number of points in the time coordinate.
	- xi,xf,nx: floats. Initial x, final x and total number of points in the x-coordinate.
	- yi,yf,ny: floats. Initial y, final y and total number of points in the y-coordinate. The xy-plane has to be big enough for the fire trajectories to fit within. Otherwise, the code will give an error.
	- delta: float. Precision in exact derivatives, f'(x)=(f(x+delta)-f(x-delta))/(2*delta).

- Initial parameters
	- ncurves: float. Number of fire trajectories to be computed.
	- nfronts: float. Number of fronts to be shown in the graphic representation.
	- steps: float. Number of steps for the ODE solver (Runge-Kutta 4).
	- point: boolean. If 'True', the initial front is a point. If 'False', it is a closed curve.
		- p0 (if True): float. Initial point.
		- frontx,fronty (if False): array. Set of points (x,y) that define the initial closed curve.
		- out (if False): boolean. If 'True', outward trajectories are computed. If 'False', inward ones are computed.

- Parameters of the model
	- z: array of dim (nx,ny). Height map that defines the surface z(x,y).
	- a: array of dim (nt,nx,ny). Parameter 'a' of the model.
	- h: array of dim (nt,nx,ny). Parameter 'h' of the model.
	- epsilon: array of dim (nt,nx,ny). Eccentricity of the ellipse in the model.
	- phi: array of dim (nt,nx,ny). Orientation of the ellipse with respect to the x-axis (on the tangent plane to the surface).

## rothermel.py
Code for Python that implements the model in the same way as model.py, with the difference that now the inputs are actual physical conditions and the parameters of the model are calculated using Rothermel's model. A thorough and complete review of this physical model can be found in the following research paper: P. L. Andrews. The Rothermel Surface Fire Spread Model and Associated Developments: A Comprehensive Explanation. Gen. Tech. Rep. RMRS-GTR-371. USDA Forest Service, Rocky Mountain Research Station, Fort Collins, 2018. Inputs (in metric units):
- Dimensions (t,x,y): the same as in model.py. Time unit: min. Space unit: m.

- Initial parameters: the same as in model.py.

- Rothermel's model
	- Fuel particle properties
		- H: array of dim (nt,nx,ny). Heat content (kJ/kg).
		- S_t: array of dim (nt,nx,ny). Total mineral content (kg minerals/kg wood).
		- S_e: array of dim (nt,nx,ny). Effective mineral content ((kg minerals-kg silica)/kg wood).
		- Rho_p: array of dim (nt,nx,ny). Particle density (kg/m**3).
	- Fuel array properties
		- Sigma: array of dim (nt,nx,ny). Surface area to volume ratio (m**2/m**3).
		- W_0: array of dim (nt,nx,ny). Fuel load (kg/m**2).
		- Delta: array of dim (nt,nx,ny). Fuel bed depth (m).
		- M_x: array of dim (nt,nx,ny). Dead fuel moisture of extinction (fraction).
	- Environmental values
		- z: array of dim (nx,ny). Surface topography (height map, m).
		- phi_s: array of dim (nx,ny). Slope steepness (angle).
		- M_f: array of dim (nt,nx,ny). Fuel moisture (kg moisture/kg wood).
		- U: array of dim (nt,nx,ny). Midflame wind speed (m/min).
		- phi_w: array of dim (nt,nx,ny). Wind direction with respect to the x-axis (on the tangent plane to the surface).

## farsite.py
Code for Python that implements the model in the case where the fire propagation is assumed to be elliptical (h=0) and the ellipse dimensions are calculated in the same way as in the Farsite simulator: M. A. Finney. FARSITE: Fire Area Simulator-model development and evaluation. Res. Pap. RMRS-RP-4, USDA Forest Service, Rocky Mountain Research Station, Ogden, 1998 (revised 2004). This simulator is also based on Rothermel's model, so the inputs are the same as in rothermel.py with the difference that Farsite uses different units. Inputs (in native units):
- Dimensions (t,x,y): the same as in model.py. Time unit: min. Space unit: ft.

- Initial parameters: the same as in model.py.

- Rothermel's model
	- Fuel particle properties
		- H: array of dim (nt,nx,ny). Heat content (Btu/lb).
		- S_t: array of dim (nt,nx,ny). Total mineral content (lb minerals/lb wood).
		- S_e: array of dim (nt,nx,ny). Effective mineral content ((lb minerals-lb silica)/lb wood).
		- Rho_p: array of dim (nt,nx,ny). Particle density (lb/ft**3).
	- Fuel array properties
		- Sigma: array of dim (nt,nx,ny). Surface area to volume ratio (ft**2/ft**3).
		- W_0: array of dim (nt,nx,ny). Fuel load (lb/ft**2).
		- Delta: array of dim (nt,nx,ny). Fuel bed depth (ft).
		- M_x: array of dim (nt,nx,ny). Dead fuel moisture of extinction (fraction).
	- Environmental values
		- z: array of dim (nx,ny). Surface topography (height map, ft).
		- phi_s: array of dim (nx,ny). Slope steepness (angle).
		- M_f: array of dim (nt,nx,ny). Fuel moisture (lb moisture/lb wood).
		- U: array of dim (nt,nx,ny). Midflame wind speed (ft/min).
		- phi_w: array of dim (nt,nx,ny). Wind direction with respect to the x-axis (on the tangent plane to the surface).
