import math

# Simple float3 implementation mirroring metal SIMD
class Vec3:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
        
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
        
    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


def simulate():
    delta_time = 0.016
    
    # 0 = block, 1 = static floor
    positions = [Vec3(0, 10, 0), Vec3(0, 0, 0)]
    predicted_pos = [Vec3(0, 10, 0), Vec3(0, 0, 0)]
    velocities = [Vec3(0, 0, 0), Vec3(0, 0, 0)]
    masses = [1.0, 0.0]  # inv_mass
    extents = [Vec3(1, 1, 1), Vec3(10, 1, 10)]
    
    print("--- Gaia Math Validation (Python Port) ---")
    print("Simulating 100 frames with dt:", delta_time)
    
    for frame in range(101):
        
        # 1. Pre-integration (Gravity)
        for i in range(2):
            if masses[i] > 0:
                velocities[i].y -= 9.81 * delta_time
                predicted_pos[i] = positions[i] + velocities[i] * delta_time
                
        # 2. XPBD Constraint Solver (AABB/Positions)
        p_a = predicted_pos[0]
        p_b = predicted_pos[1]
        
        a_min = p_a - extents[0]
        a_max = p_a + extents[0]
        b_min = p_b - extents[1]
        b_max = p_b + extents[1]
        
        overlap_x = a_min.x <= b_max.x and a_max.x >= b_min.x
        overlap_y = a_min.y <= b_max.y and a_max.y >= b_min.y
        overlap_z = a_min.z <= b_max.z and a_max.z >= b_min.z
        
        pairs = 0
        if overlap_x and overlap_y and overlap_z:
            pairs = 1
            depth = b_max.y - a_min.y
            predicted_pos[0].y += depth
            
        # 3. Velocity Derivation
        for i in range(2):
            if masses[i] > 0:
                velocities[i] = (predicted_pos[i] - positions[i]) / delta_time
                positions[i] = predicted_pos[i]
                
        if frame % 10 == 0 or pairs > 0:
            print(f"Frame {frame:03d}: Object Y = {positions[0].y:6.3f} | Vel Y = {velocities[0].y:6.3f} | Collision = {pairs}")
            if pairs > 0 and frame > 80:
                break

if __name__ == "__main__":
    simulate()
