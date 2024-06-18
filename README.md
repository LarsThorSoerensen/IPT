CREATED by Lars Thor Sørensen, (github: LarsThorSoerensen)
The Impurity particle trajectory code is to simulate the particle motion of a particle inside a fusion reactor
i.e. under influence of electromagnetic E,B fields
Was coded under a 5 ECTS special course with Anders H. Nielsen and Alexander S. Trysøe at PPFE, DTU Physics

Description of this repo:
Tungsten_IPT.pdf is the final handin from the special course
feltor.pdf is a pdf file describing the feltor code, mainly used to find a description of the feltor data

**Impurity_Particle_Trajectory folder**:

### IPT.py 
  The main code, written in python
  4 objects (classes) have been defined: Particle, ParticleSystem, f, RK4
  
  Particle contains position, velocity, time, charge, chemical symbol and mass of the simulated particle
  
  ParticleSystem is a list of Particle objects
        The original idea was to implement a method of set_position and set_velocity, so that all particles would have their position and velocity updated for a single call of the method
            Currently the position and velocity update is done by updating the Particle object
        In the future: implement some data saving method (outcommented part)
    
    The equations of motion (EoM) is put into the f object
        The E and B fields used in the Lorentz force and i.e. the EoM can be different, e.g. by interpolating on data or using some analytical function for B. Currently this is determined by the B_choice (some integer)
            Maybe this could be done more nice
            Further on, The E field was changed by changing its value in the f object, that solution is not super nice, maybe make it 0 by default, and posibly allow it to be specified as a keyword in the init or so, or maybe a set_E method
        varphi has 4 if statements, to keep the angle between [0,2pi]
        unit vectors are defined for the cylindrical coordinate system (s,phi,z) and the feltor cylindrical coordinate system (R,Z,varphi)

    RK4
        implements a Runge-Kutta-fourth order method
        currently just for at fixed timestep dt
        uses a for loop over the particle list, could maybe be made vectorized, but then the calculation in the EoM should also be changed so that it takes in a vector of N particles

        - constants and ASE
            some constants are defined, and the ase (atomic simulation environment) package is needed (ase.data) to get the atomic mass given a chemical symbol as input. If one does not have ase, one could code around this. I believe the atomic mass is averaged over isotopes, so this could also be a slight error since we only consider a single particle.
        
###
        
    
### ErrorPlot.ipynb
code used to make the performance plots in Tungsten_IPT.pdf
One can also see how to use the objects from IPT.py
    
### SanityPlots.ipynb
Code used to make the sanity check-plots in Tungsten_IPT.pdf
One can also see how to use the objects from IPT.py

### test.ipynb
Example of making final grad-B drift plot interpolating on feltor data in Tungsten_IPT.pdf
