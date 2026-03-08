#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <saltro/validation/validate_juliantime.h>

#include <limits>
#include <string>

using VecX = Eigen::VectorXd;

TEST_CASE("Valid Julian time vector passes validation", "[juliantime][validation]") {
    VecX jtime(4);
    jtime << 0.20, 0.26, 0.31, 0.40;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Empty Julian time vector fails", "[juliantime][validation]") {
    VecX jtime(0);

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime is empty");
}

TEST_CASE("Julian time vector with NaN fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.21, std::numeric_limits<double>::quiet_NaN(), 0.24;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime contains non-finite values");
}

TEST_CASE("Julian time vector with infinity fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.21, std::numeric_limits<double>::infinity(), 0.24;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime contains non-finite values");
}

TEST_CASE("Julian time vector with zero fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.21, 0.0, 0.24;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime contains zero values");
}

TEST_CASE("Julian time below mission bounds fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.19, 0.22, 0.24;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime is outside mission bounds [0.20, 0.40] Julian centuries");
}

TEST_CASE("Julian time above mission bounds fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.22, 0.24, 0.41;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime is outside mission bounds [0.20, 0.40] Julian centuries");
}

TEST_CASE("Julian time with repeated values fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.22, 0.22, 0.24;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime must be strictly increasing");
}

TEST_CASE("Julian time with decreasing values fails", "[juliantime][validation]") {
    VecX jtime(3);
    jtime << 0.24, 0.23, 0.25;

    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "jtime must be strictly increasing");
}
