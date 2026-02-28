#include <saltro/validation/validate_satellite.h>
#include <saltro/validation/validate_plannersettings.h>
#include <saltro/pybind/actuators/MTQ.h>
#include <saltro/pybind/actuators/RW.h>
#include <cmath>
#include <Eigen/Dense>

namespace saltro::validation {

bool validateSatellite(const Satellite& satellite, std::string& error_msg) {
    // ===================================================================
    // Validate inertia matrix
    // ===================================================================
    
    const Satellite::Mat33& J = satellite.inertia();
    
    // Check for finite values
    if (!J.allFinite()) {
        error_msg = "inertia matrix contains non-finite values";
        return false;
    }
    
    // Check for symmetry
    const double sym_tol = 1e-10;
    if ((J - J.transpose()).cwiseAbs().maxCoeff() > sym_tol) {
        error_msg = "inertia matrix is not symmetric";
        return false;
    }
    
    // Check positive definiteness via determinant
    double det = J.determinant();
    if (det <= 1e-12) {
        error_msg = "inertia matrix determinant is too small or non-positive";
        return false;
    }
    
    // Check positive definiteness via eigenvalues
    Eigen::SelfAdjointEigenSolver<Satellite::Mat33> eigensolver(J);
    if (eigensolver.info() != Eigen::Success) {
        error_msg = "inertia matrix eigenvalue computation failed";
        return false;
    }
    
    Satellite::Vec3 eigenvals = eigensolver.eigenvalues();
    for (int i = 0; i < 3; ++i) {
        if (eigenvals(i) <= 1e-12) {
            error_msg = "inertia matrix has non-positive eigenvalues (not positive definite)";
            return false;
        }
    }
    
    // Check reasonable magnitude (not too large, not too small)
    if (J.norm() < 1e-6 || J.norm() > 1e6) {
        error_msg = "inertia matrix magnitude out of reasonable range";
        return false;
    }
    
    // ===================================================================
    // Validate actuator counts
    // ===================================================================
    
    int num_mtq = satellite.numMTQ();
    int num_rw = satellite.numRW();
    
    if (num_mtq < 0 || num_mtq > saltro::limits::MAX_NUM_MTQ) {
        error_msg = "number of MTQs out of valid range";
        return false;
    }
    
    if (num_rw < 0 || num_rw > saltro::limits::MAX_NUM_RW) {
        error_msg = "number of RWs out of valid range";
        return false;
    }
    
    // ===================================================================
    // Validate MTQ configurations
    // ===================================================================
    
    for (int i = 0; i < num_mtq; ++i) {
        try {
            const MTQ& mtq = satellite.getMTQ(i);
            
            // Check that axis is reasonable (unit vector or normalized)
            const Satellite::Vec3& axis = mtq.axis();
            
            double axis_norm = axis.norm();
            if (!std::isfinite(axis_norm) || axis_norm < 0.9 || axis_norm > 1.1) {
                error_msg = "MTQ " + std::to_string(i) + " axis is not normalized";
                return false;
            }
            
            // u_max for MTQ represents max dipole moment
            double max_dipole = mtq.u_max();
            if (!std::isfinite(max_dipole) || max_dipole <= 0.0) {
                error_msg = "MTQ " + std::to_string(i) + " max dipole invalid";
                return false;
            }
            
            if (max_dipole > 1e6) {
                error_msg = "MTQ " + std::to_string(i) + " max dipole unreasonably large";
                return false;
            }
        } catch (const std::exception& e) {
            error_msg = std::string("MTQ ") + std::to_string(i) + " access failed: " + e.what();
            return false;
        }
    }
    
    // ===================================================================
    // Validate RW configurations
    // ===================================================================
    
    for (int i = 0; i < num_rw; ++i) {
        try {
            const RW& rw = satellite.getRW(i);
            
            // Validate axis
            const Satellite::Vec3& axis = rw.axis();
            double axis_norm = axis.norm();
            if (!std::isfinite(axis_norm) || axis_norm < 0.9 || axis_norm > 1.1) {
                error_msg = "RW " + std::to_string(i) + " axis is not normalized";
                return false;
            }
            
            // Validate max torque (u_max for RW represents max torque)
            double max_torque = rw.u_max();
            if (!std::isfinite(max_torque) || max_torque <= 0.0) {
                error_msg = "RW " + std::to_string(i) + " max torque invalid";
                return false;
            }
            
            if (max_torque > 1e4) {
                error_msg = "RW " + std::to_string(i) + " max torque unreasonably large";
                return false;
            }
            
            // Validate wheel inertia
            double J_wheel = rw.wheelInertia();
            if (!std::isfinite(J_wheel) || J_wheel <= 0.0) {
                error_msg = "RW " + std::to_string(i) + " wheel inertia invalid";
                return false;
            }
            
            if (J_wheel > 1e3) {
                error_msg = "RW " + std::to_string(i) + " wheel inertia unreasonably large";
                return false;
            }
            
            // Validate momentum bounds
            double h_init = rw.momentum();
            double h_max = rw.momentumMax();
            
            if (!std::isfinite(h_init) || !std::isfinite(h_max)) {
                error_msg = "RW " + std::to_string(i) + " momentum values not finite";
                return false;
            }
            
            if (h_max <= 0.0) {
                error_msg = "RW " + std::to_string(i) + " max momentum must be positive";
                return false;
            }
            
            if (std::abs(h_init) > h_max) {
                error_msg = "RW " + std::to_string(i) + " initial momentum exceeds max";
                return false;
            }
            
            if (h_max > 1e4) {
                error_msg = "RW " + std::to_string(i) + " max momentum unreasonably large";
                return false;
            }
        } catch (const std::exception& e) {
            error_msg = std::string("RW ") + std::to_string(i) + " access failed: " + e.what();
            return false;
        }
    }
    
    // ===================================================================
    // Validate inertia consistency for RWs
    // ===================================================================
    
    const Satellite::Mat33& J_noRW = satellite.inertiaNoRW();
    if (!J_noRW.allFinite()) {
        error_msg = "inertia without RW contains non-finite values";
        return false;
    }
    
    // inertia should be >= inertia_noRW (adding RW momentum increases effective inertia)
    for (int i = 0; i < 3; ++i) {
        if (J(i, i) < J_noRW(i, i) - 1e-10) {
            error_msg = "inertia matrix inconsistent with inertia_noRW";
            return false;
        }
    }
    
    // ===================================================================
    // Validate geometry configuration
    // ===================================================================
    
    const saltro::disturbances::GeometryConfig& geom = satellite.geometryConfig();
    size_t num_faces = geom.numFaces();
    
    if (num_faces > saltro::limits::MAX_NUM_GEOMETRY_FACES) {
        error_msg = "geometry has more faces than maximum allowed";
        return false;
    }
    
    for (size_t j = 0; j < num_faces; ++j) {
        const saltro::disturbances::GeometryFace& face = geom.getFace(j);
        
        // Check area
        if (!std::isfinite(face.area) || face.area < 0.0) {
            error_msg = "geometry face " + std::to_string(j) + " area invalid";
            return false;
        }
        
        // Check centroid is finite
        if (!face.centroid.allFinite()) {
            error_msg = "geometry face " + std::to_string(j) + " centroid not finite";
            return false;
        }
        
        // Check normal is normalized
        double normal_norm = face.normal.norm();
        if (!std::isfinite(normal_norm) || normal_norm < 0.99 || normal_norm > 1.01) {
            error_msg = "geometry face " + std::to_string(j) + " normal not normalized";
            return false;
        }
        
        // Check optical coefficients are in [0, 1]
        if (face.eta_s < 0.0 || face.eta_s > 1.0 || !std::isfinite(face.eta_s)) {
            error_msg = "geometry face " + std::to_string(j) + " specular coefficient invalid";
            return false;
        }
        
        if (face.eta_d < 0.0 || face.eta_d > 1.0 || !std::isfinite(face.eta_d)) {
            error_msg = "geometry face " + std::to_string(j) + " diffuse coefficient invalid";
            return false;
        }
        
        if (face.eta_a < 0.0 || face.eta_a > 1.0 || !std::isfinite(face.eta_a)) {
            error_msg = "geometry face " + std::to_string(j) + " absorptivity invalid";
            return false;
        }
        
        // Coefficients should sum to ~1 (energy conservation)
        double coeff_sum = face.eta_s + face.eta_d + face.eta_a;
        if (coeff_sum < 0.99 || coeff_sum > 1.01) {
            error_msg = "geometry face " + std::to_string(j) + " optical coefficients don't sum to 1";
            return false;
        }
        
        // Check drag coefficient
        if (face.CD < 0.0 || !std::isfinite(face.CD)) {
            error_msg = "geometry face " + std::to_string(j) + " drag coefficient invalid";
            return false;
        }
        
        if (face.CD > 3.0) {
            error_msg = "geometry face " + std::to_string(j) + " drag coefficient unreasonably large";
            return false;
        }
    }
    
    // ===================================================================
    // Validate planner settings (if attached)
    // ===================================================================
    
    // Note: If planner settings are stored in Satellite, validate them too
    // For now, this is optional since settings might be passed separately
    // Uncomment if Satellite stores settings:
    // std::string settings_error;
    // if (!validatePlannerSettings(satellite.settings(), settings_error)) {
    //     error_msg = "planner settings validation failed: " + settings_error;
    //     return false;
    // }
    
    return true;
}

}  // namespace saltro::validation
