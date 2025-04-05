import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import io
from IPython.display import HTML

class MembraneSimulation:
    def __init__(
        self,
        n_steps=1000,
        dt=0.001,
        box_length=10.0,
        temperature=300.0,
        tau_T=0.1,
        print_interval=100,
        random_seed=42,
        membrane_thickness=1.0,
        membrane_position=5.0,
        membrane_elasticity=50.0,
        capture_strength=10.0
    ):
        """
        Initialize a 3D membrane simulation with atom capture visualization
        
        Parameters:
        -----------
        n_steps : int
            Number of simulation steps
        dt : float
            Time step for simulation
        box_length : float
            Length of the cubic simulation box
        temperature : float
            Target temperature for the thermostat
        tau_T : float
            Coupling time for Berendsen thermostat
        print_interval : int
            Interval for printing simulation status
        random_seed : int
            Random seed for reproducibility
        membrane_thickness : float
            Thickness of the membrane
        membrane_position : float
            Z-position of the membrane center
        membrane_elasticity : float
            Elasticity coefficient of the membrane
        capture_strength : float
            Strength of the capture interaction with the membrane
        """
        self.n_steps = n_steps
        self.dt = dt
        self.box_length = box_length
        self.temperature = temperature
        self.tau_T = tau_T
        self.print_interval = print_interval
        self.random_seed = random_seed
        
        # Boltzmann constant
        self.k_B = 1.380649e-23  # J/K

        # Membrane parameters
        self.membrane_thickness = membrane_thickness
        self.membrane_position = membrane_position
        self.membrane_elasticity = membrane_elasticity
        self.capture_strength = capture_strength
        
        # Lists to store trajectory data for visualization
        self.trajectory = []
        self.captured_atoms = set()
        
        # Build system data for the demo
        self._build_demo_system()

        # Initialize velocities with Maxwell-Boltzmann distribution
        np.random.seed(self.random_seed)
        self.velocities = np.zeros((self.n_particles, 3))
        for i in range(self.n_particles):
            sigma_i = np.sqrt(self.k_B * self.temperature / self.masses[i])
            self.velocities[i] = sigma_i * np.random.randn(3)
        # Remove COM velocity
        self.velocities -= np.mean(self.velocities, axis=0)

        # Save initial positions for trajectory
        self.trajectory.append(self.positions.copy())

        # Initial forces + leapfrog half-step
        init_forces = self._compute_forces(self.positions)
        self.velocities_half = self.velocities - 0.5 * self.dt * init_forces / self.masses[:, None]

    def _build_demo_system(self):
        """
        Create a demo system with multiple atoms of different types
        """
        # We'll create a simple system with different atom types for the demo
        # Define some sample molecule parameters
        
        # Number of molecules of each type
        n_water_molecules = 10
        n_methane_molecules = 5
        n_ion_pairs = 3
        
        # Lists to build up our system
        positions = []
        atom_symbols = []
        atom_names = []
        masses = []
        charges = []
        sigmas = []
        epsilons = []
        
        # Create molecules of different types randomly positioned in the box
        # Water molecules (H2O): 3 atoms each
        for i in range(n_water_molecules):
            base_pos = np.random.rand(3) * self.box_length
            base_pos[2] = 7.0 + np.random.rand() * 2.0  # Start above the membrane
            
            # O atom
            positions.append(base_pos)
            atom_symbols.append('O')
            atom_names.append('OW')
            masses.append(16.0 * 1.660539e-27)  # amu to kg
            charges.append(-0.8)
            sigmas.append(0.3166 * 1e-9)  # nm to m
            epsilons.append(0.65 * 1000.0 / 6.02214076e23)  # kJ/mol to J
            
            # H atoms
            h1_pos = base_pos + np.array([0.1, 0.0, 0.033])
            h2_pos = base_pos + np.array([-0.033, 0.1, -0.033])
            
            positions.append(h1_pos)
            atom_symbols.append('H')
            atom_names.append('HW')
            masses.append(1.008 * 1.660539e-27)
            charges.append(0.4)
            sigmas.append(0.0 * 1e-9)
            epsilons.append(0.0 * 1000.0 / 6.02214076e23)
            
            positions.append(h2_pos)
            atom_symbols.append('H')
            atom_names.append('HW')
            masses.append(1.008 * 1.660539e-27)
            charges.append(0.4)
            sigmas.append(0.0 * 1e-9)
            epsilons.append(0.0 * 1000.0 / 6.02214076e23)
        
        # Methane molecules (CH4): 5 atoms each
        for i in range(n_methane_molecules):
            base_pos = np.random.rand(3) * self.box_length
            base_pos[2] = 7.0 + np.random.rand() * 2.0  # Start above the membrane
            
            # C atom
            positions.append(base_pos)
            atom_symbols.append('C')
            atom_names.append('C')
            masses.append(12.011 * 1.660539e-27)
            charges.append(0.0)
            sigmas.append(0.34 * 1e-9)
            epsilons.append(0.46 * 1000.0 / 6.02214076e23)
            
            # H atoms in tetrahedral arrangement
            h_displace = [
                np.array([0.1, 0.1, 0.1]),
                np.array([-0.1, -0.1, 0.1]),
                np.array([0.1, -0.1, -0.1]),
                np.array([-0.1, 0.1, -0.1])
            ]
            
            for h_disp in h_displace:
                h_pos = base_pos + h_disp
                positions.append(h_pos)
                atom_symbols.append('H')
                atom_names.append('H')
                masses.append(1.008 * 1.660539e-27)
                charges.append(0.0)
                sigmas.append(0.25 * 1e-9)
                epsilons.append(0.13 * 1000.0 / 6.02214076e23)
        
        # Ion pairs (Na+ and Cl-)
        for i in range(n_ion_pairs):
            # Sodium ion
            na_pos = np.random.rand(3) * self.box_length
            na_pos[2] = 8.0 + np.random.rand() * 1.0  # Start above the membrane
            
            positions.append(na_pos)
            atom_symbols.append('Na')
            atom_names.append('Na+')
            masses.append(22.99 * 1.660539e-27)
            charges.append(1.0)
            sigmas.append(0.33 * 1e-9)
            epsilons.append(0.012 * 1000.0 / 6.02214076e23)
            
            # Chloride ion
            cl_pos = np.random.rand(3) * self.box_length
            cl_pos[2] = 8.0 + np.random.rand() * 1.0
            
            positions.append(cl_pos)
            atom_symbols.append('Cl')
            atom_names.append('Cl-')
            masses.append(35.45 * 1.660539e-27)
            charges.append(-1.0)
            sigmas.append(0.44 * 1e-9)
            epsilons.append(0.46 * 1000.0 / 6.02214076e23)
        
        # Convert lists to numpy arrays
        self.positions = np.array(positions)
        self.atom_symbols = atom_symbols
        self.atom_names = atom_names
        self.masses = np.array(masses)
        self.charges = np.array(charges)
        self.sigmas = np.array(sigmas)
        self.epsilons = np.array(epsilons)
        self.atom_colors = self._assign_atom_colors()
        
        self.n_particles = len(self.masses)
        
        # Ensure all positions are within box
        self.positions %= self.box_length

    def _assign_atom_colors(self):
        """Assign colors to atoms based on their element symbol"""
        color_map = {
            'C': 'black',
            'H': 'lightgray',
            'O': 'red',
            'N': 'blue',
            'S': 'yellow',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'brown',
            'I': 'purple',
            'Na': 'purple'
        }
        # Default color for any unknown element
        default_color = 'gray'
        return [color_map.get(symbol, default_color) for symbol in self.atom_symbols]

    def _compute_forces(self, positions):
        """
        Compute forces including:
        1. Lennard-Jones interactions between atoms
        2. Membrane interaction forces
        """
        forces = np.zeros_like(positions)
        
        # 1. Lennard-Jones forces
        for i in range(self.n_particles):
            for j in range(i+1, self.n_particles):
                rij = positions[i] - positions[j]
                # PBC: apply minimal image
                rij -= self.box_length * np.round(rij / self.box_length)
                r2 = np.dot(rij, rij)
                if r2 == 0.0:
                    continue

                # Lorentz-Berthelot mixing rules
                sigma_ij = 0.5*(self.sigmas[i] + self.sigmas[j])
                epsilon_ij = np.sqrt(self.epsilons[i] * self.epsilons[j])

                cutoff = 3.0*sigma_ij
                if r2 < cutoff**2:
                    inv_r2 = 1.0 / r2
                    r6 = inv_r2**3
                    sig2 = sigma_ij**2
                    sig6_r6 = (sig2*inv_r2)**3
                    prefactor = 24.0*epsilon_ij*r6
                    term = 2.0*(sig6_r6**2) - sig6_r6
                    f_scalar = prefactor*term
                    fij = f_scalar*rij
                    forces[i] += fij
                    forces[j] -= fij
        
        # 2. Membrane interaction forces
        self._add_membrane_forces(positions, forces)
        
        # 3. Simple gravity-like force to help atoms fall toward membrane
        gravity = np.zeros_like(positions)
        gravity[:, 2] = -1e-12  # Very small downward force
        forces += gravity
        
        return forces

    def _add_membrane_forces(self, positions, forces):
        """
        Add forces due to membrane interaction
        Creates a membrane-like barrier with capture abilities
        """
        half_thickness = self.membrane_thickness / 2.0
        membrane_min = self.membrane_position - half_thickness
        membrane_max = self.membrane_position + half_thickness
        
        for i in range(self.n_particles):
            z = positions[i, 2]
            
            # Check if the atom is within the membrane region
            if membrane_min <= z <= membrane_max:
                # Atom is inside the membrane - apply capturing forces
                # Strong harmonic potential to trap the atom at the membrane center
                dz = z - self.membrane_position
                k_capture = self.capture_strength
                forces[i, 2] -= k_capture * dz
                
                # Mark this atom as captured
                self.captured_atoms.add(i)
            else:
                # If atom is approaching the membrane, apply an elastic barrier
                # that gets stronger as it gets closer
                if z < membrane_min:
                    # Coming from below
                    dz = z - membrane_min
                    k_elastic = self.membrane_elasticity * np.exp(-abs(dz))
                    forces[i, 2] -= k_elastic * dz
                elif z > membrane_max:
                    # Coming from above
                    dz = z - membrane_max
                    k_elastic = self.membrane_elasticity * np.exp(-abs(dz))
                    forces[i, 2] -= k_elastic * dz
    
    def _kinetic_energy(self, velocities):
        """Calculate the kinetic energy of the system"""
        return 0.5 * np.sum(self.masses * np.sum(velocities**2, axis=1))

    def _berendsen_thermostat(self, velocities, current_temp):
        """Apply Berendsen thermostat to control temperature"""
        if current_temp <= 0.0:
            return velocities
        lam = np.sqrt(1.0 + (self.dt/self.tau_T)*(self.temperature/current_temp - 1.0))
        return velocities * lam

    def run(self):
        """
        Run the simulation for n_steps and store trajectory data
        Returns the positions and velocities at the end of the simulation
        """
        for step in range(self.n_steps):
            # 1) Half-step velocity update
            forces = self._compute_forces(self.positions)
            self.velocities_half += 0.5*self.dt * (forces / self.masses[:, None])

            # 2) Update positions
            self.positions += self.dt*self.velocities_half
            self.positions %= self.box_length  # PBC

            # 3) Next forces
            new_forces = self._compute_forces(self.positions)

            # 4) Next half-step
            self.velocities_half += 0.5*self.dt * (new_forces / self.masses[:, None])

            # 5) Full-step velocity
            velocities_full = self.velocities_half + 0.5*self.dt * (new_forces / self.masses[:, None])

            # 6) Berendsen thermostat
            E_kin = self._kinetic_energy(velocities_full)
            dof = 3*self.n_particles
            current_temp = (2.0*E_kin)/(dof*self.k_B)
            velocities_full = self._berendsen_thermostat(velocities_full, current_temp)

            # Shift back to half-step
            self.velocities_half = velocities_full - 0.5*self.dt*(new_forces / self.masses[:, None])

            # Store position for trajectory
            self.trajectory.append(self.positions.copy())

            if step % self.print_interval == 0:
                captured_count = len(self.captured_atoms)
                print(f"Step {step:5d} | T = {current_temp:.2f} K | Captured: {captured_count}/{self.n_particles}")

        final_velocities = self.velocities_half + 0.5*self.dt*(new_forces / self.masses[:, None])
        print("Simulation complete.")
        return self.positions, final_velocities
    
    def visualize_trajectory(self, skip_frames=10):
        """
        Create a 3D animation of the simulation trajectory
        
        Parameters:
        -----------
        skip_frames : int
            Number of frames to skip for faster animation
        """
        # Convert trajectory to numpy array
        trajectory = np.array(self.trajectory[::skip_frames])
        n_frames = len(trajectory)
        
        # Determine the frames where each atom is captured
        capture_frames = {}
        for frame_idx in range(n_frames):
            frame = trajectory[frame_idx]
            membrane_min = self.membrane_position - self.membrane_thickness/2
            membrane_max = self.membrane_position + self.membrane_thickness/2
            
            for atom_idx in range(self.n_particles):
                z = frame[atom_idx, 2]
                if membrane_min <= z <= membrane_max:
                    if atom_idx not in capture_frames:
                        capture_frames[atom_idx] = frame_idx
        
        # Initialize 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        ax.set_xlim(0, self.box_length)
        ax.set_ylim(0, self.box_length)
        ax.set_zlim(0, self.box_length)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_zlabel('Z (nm)')
        ax.set_title('Membrane Capturing Atoms Simulation')
        
        # Add opacity to the membrane for better visualization
        membrane_min = self.membrane_position - self.membrane_thickness/2
        membrane_max = self.membrane_position + self.membrane_thickness/2
        
        # Creating a semi-transparent membrane representation
        membrane_x = np.linspace(0, self.box_length, 10)
        membrane_y = np.linspace(0, self.box_length, 10)
        X, Y = np.meshgrid(membrane_x, membrane_y)
        Z_bottom = np.ones_like(X) * membrane_min
        Z_top = np.ones_like(X) * membrane_max
        
        # Plot the membrane
        membrane_alpha = 0.3
        membrane_color = 'skyblue'
        ax.plot_surface(X, Y, Z_bottom, alpha=membrane_alpha, color=membrane_color, shade=False)
        ax.plot_surface(X, Y, Z_top, alpha=membrane_alpha, color=membrane_color, shade=False)
        
        # Create scatter plot for atoms
        scatter = ax.scatter(
            trajectory[0, :, 0], 
            trajectory[0, :, 1], 
            trajectory[0, :, 2],
            c=self.atom_colors,
            s=50 * np.array([1.0 if i in self.captured_atoms else 0.7 for i in range(self.n_particles)]),
            alpha=0.8
        )
        
        # Text annotation for captured atoms count
        captured_text = ax.text2D(0.05, 0.95, f"Captured: 0/{self.n_particles}", 
                                transform=ax.transAxes, fontsize=12)
        
        # Time indicator
        time_text = ax.text2D(0.05, 0.9, f"Time: 0.0 ps", transform=ax.transAxes, fontsize=12)
        
        def update(frame):
            # Update positions
            scatter._offsets3d = (
                trajectory[frame, :, 0],
                trajectory[frame, :, 1],
                trajectory[frame, :, 2]
            )
            
            # Count captured atoms at this frame
            captured_count = sum(1 for i in range(self.n_particles) 
                               if membrane_min <= trajectory[frame, i, 2] <= membrane_max)
            
            # Update sizes: larger for captured atoms
            sizes = []
            for i in range(self.n_particles):
                # If atom is captured in this frame
                if membrane_min <= trajectory[frame, i, 2] <= membrane_max:
                    sizes.append(80)  # Larger size for captured atoms
                # If atom was captured before
                elif i in capture_frames and frame >= capture_frames[i]:
                    # Atom was captured earlier but might have escaped
                    sizes.append(65)
                else:
                    sizes.append(40)  # Normal size for free atoms
                    
            scatter.set_sizes(sizes)
            
            # Update text
            captured_text.set_text(f"Captured: {captured_count}/{self.n_particles}")
            time_text.set_text(f"Time: {frame * skip_frames * self.dt:.2f} ps")
            
            return scatter, captured_text, time_text
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
        
        plt.tight_layout()
        
        return ani, fig

# Run a demo simulation
def run_demo():
    print("Starting membrane simulation demo...")
    
    # Create simulation with parameters for a good visual demo
    sim = MembraneSimulation(
        n_steps=500,           # Number of simulation steps
        dt=0.002,              # Time step
        box_length=10.0,       # Box size
        temperature=350.0,     # Higher temperature for more movement
        membrane_thickness=1.5, # Thicker membrane for better visibility
        membrane_position=5.0, # Membrane in middle of box
        membrane_elasticity=30.0, # Elasticity of membrane
        capture_strength=8.0,  # Strength of capture
        print_interval=50      # Print interval
    )
    
    # Run the simulation
    sim.run()
    
    print("\nGenerating visualization...")
    
    # Create the visualization
    ani, fig = sim.visualize_trajectory(skip_frames=2)
    
    print("Simulation complete!")
    
    # Return the animation and figure
    return ani, fig, sim

# Uncomment to run the demo
ani, fig, sim = run_demo()
plt.show()
