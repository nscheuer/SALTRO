#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>
#include <array>
#include <cstddef>

namespace saltro::disturbances {

/**
 * @brief Represents a single geometric face/surface element of a satellite.
 * 
 * Each face has:
 * - Geometric properties: area, centroid, normal vector
 * - Optical properties: specular (eta_s), diffuse (eta_d), absorptivity (eta_a)
 * - Aerodynamic properties: drag coefficient (CD)
 */
struct GeometryFace {
    using Vec3 = Eigen::Vector3d;

    double area;        ///< Surface area [m²]
    Vec3 centroid;      ///< Centroid position in body frame [m]
    Vec3 normal;        ///< Surface normal unit vector in body frame
    double eta_s;       ///< Specular reflection coefficient [0-1]
    double eta_d;       ///< Diffuse reflection coefficient [0-1]
    double eta_a;       ///< Absorptivity coefficient [0-1]
    double CD;          ///< Drag coefficient

    /**
     * @brief Default constructor initializes all values to zero.
     */
    GeometryFace();

    /**
     * @brief Construct a GeometryFace with specified parameters.
     * 
     * @param area_ Surface area [m²]
     * @param centroid_ Centroid position vector [m]
     * @param normal_ Surface normal unit vector
     * @param eta_s_ Specular reflection coefficient (default: 0)
     * @param eta_d_ Diffuse reflection coefficient (default: 0)
     * @param eta_a_ Absorptivity coefficient (default: 0)
     * @param CD_ Drag coefficient (default: 0)
     */
    GeometryFace(double area_, const Vec3& centroid_, const Vec3& normal_,
                 double eta_s_ = 0.0, double eta_d_ = 0.0, 
                 double eta_a_ = 0.0, double CD_ = 0.0);
};

/**
 * @brief Static, flight-safe configuration for satellite geometry.
 * 
 * The GeometryConfig class defines the geometric and aerodynamic properties
 * of the satellite's external surfaces. This configuration is used by
 * disturbance models (e.g., drag, solar radiation pressure) to compute
 * forces and torques.
 * 
 * This implementation uses static allocation with a maximum number of faces
 * defined by saltro::limits::MAX_NUM_GEOMETRY_FACES, ensuring no dynamic
 * memory allocation occurs during runtime (flight-safe).
 */
class GeometryConfig {
public:
    using Vec3 = Eigen::Vector3d;
    using FaceArray = std::array<GeometryFace, saltro::limits::MAX_NUM_GEOMETRY_FACES>;

    /**
     * @brief Default constructor creates an empty geometry configuration.
     */
    GeometryConfig();

    /**
     * @brief Add a face to the geometry configuration.
     * 
     * @param face The GeometryFace to add
     * @return true if face was added successfully, false if max capacity reached
     */
    bool addFace(const GeometryFace& face);

    /**
     * @brief Get the number of faces currently configured.
     * 
     * @return Number of faces
     */
    size_t numFaces() const { return num_faces_; }

    /**
     * @brief Get a reference to a specific face.
     * 
     * @param index Face index [0, numFaces())
     * @return Reference to the GeometryFace
     */
    const GeometryFace& getFace(size_t index) const;

    /**
     * @brief Get mutable reference to a specific face.
     * 
     * @param index Face index [0, numFaces())
     * @return Reference to the GeometryFace
     */
    GeometryFace& getFace(size_t index);

    /**
     * @brief Clear all faces from the configuration.
     */
    void clear();

    /**
     * @brief Get the maximum number of faces that can be stored.
     * 
     * @return Maximum capacity
     */
    static constexpr size_t maxFaces() { return saltro::limits::MAX_NUM_GEOMETRY_FACES; }

    /**
     * @brief Direct access to the underlying face array (const).
     * 
     * @return Const reference to the face array
     */
    const FaceArray& faces() const { return faces_; }

    /**
     * @brief Direct access to the underlying face array (mutable).
     * 
     * @return Reference to the face array
     */
    FaceArray& faces() { return faces_; }

private:
    FaceArray faces_;       ///< Static array of geometry faces
    size_t num_faces_;      ///< Current number of active faces
};

}  // namespace saltro::disturbances
