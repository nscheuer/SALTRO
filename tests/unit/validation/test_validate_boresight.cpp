#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/validation/validate_boresight.h>

#include <cmath>
#include <limits>
#include <string>

using MatX = Eigen::MatrixXd;

static MatX validBoresightHistory() {
    MatX boresight(3, 2);
    boresight.col(0) << 1.0, 0.0, 0.0;
    boresight.col(1) << 0.0, 1.0, 0.0;
    return boresight;
}

TEST_CASE("Valid boresight history passes validation", "[boresight][validation]") {
    const MatX boresight = validBoresightHistory();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid single column boresight passes validation", "[boresight][validation]") {
    MatX boresight(3, 1);
    boresight.col(0) << 0.0, 0.0, 1.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid multiple columns with normalized vectors pass", "[boresight][validation]") {
    MatX boresight(3, 4);
    boresight.col(0) << 1.0, 0.0, 0.0;
    boresight.col(1) << 0.0, 1.0, 0.0;
    boresight.col(2) << 0.0, 0.0, 1.0;
    boresight.col(3) << 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0);

    std::string error_msg;
    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Invalid row count fails", "[boresight][validation]") {
    MatX boresight(2, 2);
    boresight.setZero();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight must have shape (3, N)");
}

TEST_CASE("Invalid row count (4 rows) fails", "[boresight][validation]") {
    MatX boresight(4, 2);
    boresight.setZero();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight must have shape (3, N)");
}

TEST_CASE("Empty boresight fails", "[boresight][validation]") {
    MatX boresight(3, 0);
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight must have at least one column");
}

TEST_CASE("Boresight exceeding MAX_LENGTH_TRAJ fails", "[boresight][validation]") {
    MatX boresight(3, saltro::limits::MAX_LENGTH_TRAJ + 1);
    boresight.row(0).setOnes();
    boresight.row(1).setZero();
    boresight.row(2).setZero();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight exceeds MAX_LENGTH_TRAJ");
}

TEST_CASE("NaN in boresight is rejected", "[boresight][validation]") {
    MatX boresight = validBoresightHistory();
    boresight(2, 1) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight column 1 contains NaN or Inf");
}

TEST_CASE("Infinity in boresight is rejected", "[boresight][validation]") {
    MatX boresight = validBoresightHistory();
    boresight(0, 0) = std::numeric_limits<double>::infinity();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight column 0 contains NaN or Inf");
}

TEST_CASE("Zero norm boresight is rejected", "[boresight][validation]") {
    MatX boresight(3, 1);
    boresight.col(0) << 0.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight column 0 has zero or non-finite norm");
}

TEST_CASE("Boresight not normalized is rejected", "[boresight][validation]") {
    MatX boresight(3, 1);
    boresight.col(0) << 2.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg.find("boresight column 0 is not normalized") == 0);
}

TEST_CASE("Boresight norm at tolerance boundary passes", "[boresight][validation][boundary]") {
    MatX boresight(3, 1);
    boresight.col(0) << 1.001, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE(ok);
}

TEST_CASE("Boresight norm beyond tolerance fails", "[boresight][validation][boundary]") {
    MatX boresight(3, 1);
    boresight.col(0) << 1.0011, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg.find("boresight column 0 is not normalized") == 0);
}

TEST_CASE("Error pinpoints failing column in set", "[boresight][validation]") {
    MatX boresight(3, 3);
    boresight.col(0) << 1.0, 0.0, 0.0;
    boresight.col(1) << 0.0, 1.0, 0.0;
    boresight.col(2) << 0.0, 0.0, 0.0;

    std::string error_msg;
    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight column 2 has zero or non-finite norm");
}

TEST_CASE("Negative infinity is rejected", "[boresight][validation]") {
    MatX boresight = validBoresightHistory();
    boresight(1, 0) = -std::numeric_limits<double>::infinity();
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "boresight column 0 contains NaN or Inf");
}

TEST_CASE("Very small but normalized vector passes", "[boresight][validation]") {
    MatX boresight(3, 1);
    const double val = 1.0 / std::sqrt(3.0);
    boresight.col(0) << val, val, val;
    std::string error_msg;

    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);

    REQUIRE(ok);
}
