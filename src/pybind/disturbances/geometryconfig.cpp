#include <saltro/pybind/disturbances/geometryconfig.h>
#include <stdexcept>

namespace saltro::disturbances {

// ============================================================================
// GeometryFace Implementation
// ============================================================================

GeometryFace::GeometryFace()
    : area(0.0),
      centroid(Vec3::Zero()),
      normal(Vec3::Zero()),
      eta_s(0.0),
      eta_d(0.0),
      eta_a(0.0),
      CD(0.0) {
}

GeometryFace::GeometryFace(double area_, const Vec3& centroid_, const Vec3& normal_,
                           double eta_s_, double eta_d_, double eta_a_, double CD_)
    : area(area_),
      centroid(centroid_),
      normal(normal_),
      eta_s(eta_s_),
      eta_d(eta_d_),
      eta_a(eta_a_),
      CD(CD_) {
}

// ============================================================================
// GeometryConfig Implementation
// ============================================================================

GeometryConfig::GeometryConfig()
    : faces_(),
      num_faces_(0) {
}

bool GeometryConfig::addFace(const GeometryFace& face) {
    if (num_faces_ >= saltro::limits::MAX_NUM_GEOMETRY_FACES) {
        return false;  // Cannot add more faces - capacity reached
    }
    
    faces_[num_faces_] = face;
    ++num_faces_;
    return true;
}

const GeometryFace& GeometryConfig::getFace(size_t index) const {
    if (index >= num_faces_) {
        throw std::out_of_range("GeometryConfig::getFace: index out of range");
    }
    return faces_[index];
}

GeometryFace& GeometryConfig::getFace(size_t index) {
    if (index >= num_faces_) {
        throw std::out_of_range("GeometryConfig::getFace: index out of range");
    }
    return faces_[index];
}

void GeometryConfig::clear() {
    num_faces_ = 0;
    // Optionally zero out the array (for security/determinism)
    // faces_.fill(GeometryFace());
}

}  // namespace saltro::disturbances
