#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/geometryconfig.h>
#include <saltro/limits.h>

using namespace saltro::disturbances;
using Vec3 = Eigen::Vector3d;

// ============================================================================
// GeometryFace Tests
// ============================================================================

TEST_CASE("GeometryFace default constructor", "[geometryface]") {
    GeometryFace face;
    
    REQUIRE(face.area == 0.0);
    REQUIRE(face.centroid.isZero());
    REQUIRE(face.normal.isZero());
    REQUIRE(face.eta_s == 0.0);
    REQUIRE(face.eta_d == 0.0);
    REQUIRE(face.eta_a == 0.0);
    REQUIRE(face.CD == 0.0);
}

TEST_CASE("GeometryFace constructor with required parameters", "[geometryface]") {
    double area = 0.1;
    Vec3 centroid(0.5, 0.0, 0.0);
    Vec3 normal(1.0, 0.0, 0.0);
    
    GeometryFace face(area, centroid, normal);
    
    REQUIRE(face.area == area);
    REQUIRE(face.centroid.isApprox(centroid));
    REQUIRE(face.normal.isApprox(normal));
    REQUIRE(face.eta_s == 0.0);
    REQUIRE(face.eta_d == 0.0);
    REQUIRE(face.eta_a == 0.0);
    REQUIRE(face.CD == 0.0);
}

TEST_CASE("GeometryFace constructor with all parameters", "[geometryface]") {
    double area = 0.1;
    Vec3 centroid(0.5, 0.0, 0.0);
    Vec3 normal(1.0, 0.0, 0.0);
    double eta_s = 0.1;
    double eta_d = 0.5;
    double eta_a = 0.4;
    double CD = 2.2;
    
    GeometryFace face(area, centroid, normal, eta_s, eta_d, eta_a, CD);
    
    REQUIRE(face.area == area);
    REQUIRE(face.centroid.isApprox(centroid));
    REQUIRE(face.normal.isApprox(normal));
    REQUIRE(face.eta_s == eta_s);
    REQUIRE(face.eta_d == eta_d);
    REQUIRE(face.eta_a == eta_a);
    REQUIRE(face.CD == CD);
}

TEST_CASE("GeometryFace attributes are mutable", "[geometryface]") {
    GeometryFace face;
    
    // Modify all attributes
    face.area = 0.5;
    face.centroid = Vec3(1.0, 2.0, 3.0);
    face.normal = Vec3(0.0, 1.0, 0.0);
    face.eta_s = 0.2;
    face.eta_d = 0.3;
    face.eta_a = 0.5;
    face.CD = 2.5;
    
    // Verify changes
    REQUIRE(face.area == 0.5);
    REQUIRE(face.centroid.isApprox(Vec3(1.0, 2.0, 3.0)));
    REQUIRE(face.normal.isApprox(Vec3(0.0, 1.0, 0.0)));
    REQUIRE(face.eta_s == 0.2);
    REQUIRE(face.eta_d == 0.3);
    REQUIRE(face.eta_a == 0.5);
    REQUIRE(face.CD == 2.5);
}

// ============================================================================
// GeometryConfig Tests
// ============================================================================

TEST_CASE("GeometryConfig default constructor", "[geometryconfig]") {
    GeometryConfig config;
    
    REQUIRE(config.numFaces() == 0);
}

TEST_CASE("GeometryConfig add single face", "[geometryconfig]") {
    GeometryConfig config;
    GeometryFace face(0.1, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0));
    
    bool result = config.addFace(face);
    
    REQUIRE(result == true);
    REQUIRE(config.numFaces() == 1);
}

TEST_CASE("GeometryConfig add multiple faces", "[geometryconfig]") {
    GeometryConfig config;
    
    for (int i = 0; i < 5; ++i) {
        Vec3 normal = Vec3::Zero();
        normal(i % 3) = 1.0;
        GeometryFace face(0.1 * (i + 1), Vec3::Zero(), normal);
        bool result = config.addFace(face);
        REQUIRE(result == true);
    }
    
    REQUIRE(config.numFaces() == 5);
}

TEST_CASE("GeometryConfig add face up to maximum", "[geometryconfig]") {
    GeometryConfig config;
    size_t max_faces = GeometryConfig::maxFaces();
    
    // Should be able to add max_faces
    for (size_t i = 0; i < max_faces; ++i) {
        GeometryFace face(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0));
        bool result = config.addFace(face);
        REQUIRE(result == true);
    }
    
    REQUIRE(config.numFaces() == max_faces);
}

TEST_CASE("GeometryConfig add face beyond maximum", "[geometryconfig]") {
    GeometryConfig config;
    size_t max_faces = GeometryConfig::maxFaces();
    
    // Fill to capacity
    for (size_t i = 0; i < max_faces; ++i) {
        GeometryFace face(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0));
        config.addFace(face);
    }
    
    // Try to add one more - should return false
    GeometryFace extra_face(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0));
    bool result = config.addFace(extra_face);
    
    REQUIRE(result == false);
    REQUIRE(config.numFaces() == max_faces);
}

TEST_CASE("GeometryConfig get face with valid index", "[geometryconfig]") {
    GeometryConfig config;
    
    double area = 0.5;
    Vec3 centroid(1.0, 2.0, 3.0);
    Vec3 normal(0.0, 0.0, 1.0);
    GeometryFace face(area, centroid, normal, 0.1, 0.2, 0.3, 2.2);
    
    config.addFace(face);
    
    const GeometryFace& retrieved_face = config.getFace(0);
    
    REQUIRE(retrieved_face.area == area);
    REQUIRE(retrieved_face.centroid.isApprox(centroid));
    REQUIRE(retrieved_face.normal.isApprox(normal));
    REQUIRE(retrieved_face.eta_s == 0.1);
    REQUIRE(retrieved_face.eta_d == 0.2);
    REQUIRE(retrieved_face.eta_a == 0.3);
    REQUIRE(retrieved_face.CD == 2.2);
}

TEST_CASE("GeometryConfig get face with invalid index", "[geometryconfig]") {
    GeometryConfig config;
    config.addFace(GeometryFace(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0)));
    
    // Index too large
    REQUIRE_THROWS_AS(config.getFace(1), std::out_of_range);
    REQUIRE_THROWS_AS(config.getFace(10), std::out_of_range);
}

TEST_CASE("GeometryConfig get face when empty", "[geometryconfig]") {
    GeometryConfig config;
    
    REQUIRE_THROWS_AS(config.getFace(0), std::out_of_range);
}

TEST_CASE("GeometryConfig clear", "[geometryconfig]") {
    GeometryConfig config;
    
    // Add some faces
    for (int i = 0; i < 5; ++i) {
        GeometryFace face(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0));
        config.addFace(face);
    }
    
    REQUIRE(config.numFaces() == 5);
    
    // Clear
    config.clear();
    
    REQUIRE(config.numFaces() == 0);
}

TEST_CASE("GeometryConfig clear allows adding again", "[geometryconfig]") {
    GeometryConfig config;
    
    // Add and clear
    config.addFace(GeometryFace(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0)));
    config.clear();
    
    // Add again
    GeometryFace face(0.2, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0));
    bool result = config.addFace(face);
    
    REQUIRE(result == true);
    REQUIRE(config.numFaces() == 1);
}

TEST_CASE("GeometryConfig max faces constant", "[geometryconfig]") {
    size_t max_faces = GeometryConfig::maxFaces();
    
    // Should be MAX_NUM_GEOMETRY_FACES from limits
    REQUIRE(max_faces == saltro::limits::MAX_NUM_GEOMETRY_FACES);
    REQUIRE(max_faces > 0);
}

TEST_CASE("GeometryConfig modify face through reference", "[geometryconfig]") {
    GeometryConfig config;
    
    GeometryFace face(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0));
    config.addFace(face);
    
    // Get reference and modify
    GeometryFace& retrieved_face = config.getFace(0);
    retrieved_face.area = 0.5;
    retrieved_face.eta_s = 0.8;
    
    // Verify changes persist
    REQUIRE(config.getFace(0).area == 0.5);
    REQUIRE(config.getFace(0).eta_s == 0.8);
}

TEST_CASE("GeometryConfig different face configurations", "[geometryconfig]") {
    GeometryConfig config;
    
    // Face 1: Front face
    GeometryFace face1(
        0.04,  // 20cm x 20cm
        Vec3(0.1, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        0.1,  // eta_s
        0.5,  // eta_d
        0.4,  // eta_a
        2.2   // CD
    );
    
    // Face 2: Solar panel
    GeometryFace face2(
        0.2,
        Vec3(0.0, 0.5, 0.0),
        Vec3(0.0, 1.0, 0.0),
        0.8,  // Highly reflective
        0.1,
        0.1,
        1.5
    );
    
    config.addFace(face1);
    config.addFace(face2);
    
    REQUIRE(config.numFaces() == 2);
    REQUIRE(config.getFace(0).area == 0.04);
    REQUIRE(config.getFace(1).area == 0.2);
    REQUIRE(config.getFace(0).eta_s == 0.1);
    REQUIRE(config.getFace(1).eta_s == 0.8);
}

TEST_CASE("GeometryConfig const and non-const getFace", "[geometryconfig]") {
    GeometryConfig config;
    config.addFace(GeometryFace(0.1, Vec3::Zero(), Vec3(1.0, 0.0, 0.0)));
    
    // Non-const access
    GeometryFace& face_mut = config.getFace(0);
    face_mut.area = 0.5;
    
    // Const access
    const GeometryConfig& config_const = config;
    const GeometryFace& face_const = config_const.getFace(0);
    
    REQUIRE(face_const.area == 0.5);
}

TEST_CASE("GeometryConfig faces array access", "[geometryconfig]") {
    GeometryConfig config;
    
    for (int i = 0; i < 3; ++i) {
        config.addFace(GeometryFace(0.1 * (i + 1), Vec3::Zero(), Vec3(1.0, 0.0, 0.0)));
    }
    
    const auto& faces = config.faces();
    
    REQUIRE_THAT(faces[0].area, Catch::Matchers::WithinRel(0.1, 1e-10));
    REQUIRE_THAT(faces[1].area, Catch::Matchers::WithinRel(0.2, 1e-10));
    REQUIRE_THAT(faces[2].area, Catch::Matchers::WithinRel(0.3, 1e-10));
}
