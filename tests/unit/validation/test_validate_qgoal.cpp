#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/validation/validate_qgoal.h>

#include <cmath>
#include <limits>
#include <string>

using MatX = Eigen::MatrixXd;

static MatX validQuaternionGoals() {
    MatX q_goal(4, 2);
    q_goal.col(0) << 1.0, 0.0, 0.0, 0.0;
    q_goal.col(1) << 0.0, 1.0, 0.0, 0.0;
    return q_goal;
}

static MatX validECIGoals() {
    MatX q_goal(4, 2);
    q_goal.col(0) << std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0, 0.0;
    q_goal.col(1) << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0;
    return q_goal;
}

TEST_CASE("Valid quaternion goals pass validation", "[qgoal][validation]") {
    const MatX q_goal = validQuaternionGoals();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid ECI goals pass validation", "[qgoal][validation]") {
    const MatX q_goal = validECIGoals();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid mixed quaternion and ECI goals pass validation", "[qgoal][validation]") {
    MatX q_goal(4, 4);
    q_goal.col(0) << 1.0, 0.0, 0.0, 0.0;
    q_goal.col(1) << std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0, 0.0;
    q_goal.col(2) << 0.0, 0.0, 0.0, 1.0;
    q_goal.col(3) << std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0, 0.0;

    std::string error_msg;
    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Invalid row count fails", "[qgoal][validation]") {
    MatX q_goal(3, 2);
    q_goal.setZero();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal must have shape (4, N)");
}

TEST_CASE("Empty q_goal fails", "[qgoal][validation]") {
    MatX q_goal(4, 0);
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal must have at least one column");
}

TEST_CASE("q_goal exceeding MAX_LENGTH_TRAJ fails", "[qgoal][validation]") {
    MatX q_goal(4, saltro::limits::MAX_LENGTH_TRAJ + 1);
    q_goal.setZero();
    q_goal.row(0).setOnes();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal exceeds MAX_LENGTH_TRAJ");
}

TEST_CASE("NaN in row 2-4 is rejected", "[qgoal][validation]") {
    MatX q_goal = validQuaternionGoals();
    q_goal(2, 1) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 1 has invalid NaN/Inf in rows 2-4");
}

TEST_CASE("Infinity in row 2-4 is rejected", "[qgoal][validation]") {
    MatX q_goal = validECIGoals();
    q_goal(3, 0) = std::numeric_limits<double>::infinity();
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 has invalid NaN/Inf in rows 2-4");
}

TEST_CASE("Zero direction norm is rejected", "[qgoal][validation]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 has zero or non-finite direction norm");
}

TEST_CASE("ECI direction not normalized is rejected", "[qgoal][validation]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << std::numeric_limits<double>::quiet_NaN(), 2.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 ECI direction is not normalized");
}

TEST_CASE("Invalid q0 value is rejected", "[qgoal][validation]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << std::numeric_limits<double>::infinity(), 1.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 has invalid q0 value");
}

TEST_CASE("Quaternion not normalized is rejected", "[qgoal][validation]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << 1.0, 1.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 quaternion is not normalized");
}

TEST_CASE("Quaternion norm at tolerance boundary passes", "[qgoal][validation][boundary]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << 1.001, 0.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE(ok);
}

TEST_CASE("Quaternion norm beyond tolerance fails", "[qgoal][validation][boundary]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << 1.0011, 0.0, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 quaternion is not normalized");
}

TEST_CASE("ECI norm at tolerance boundary passes", "[qgoal][validation][boundary]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << std::numeric_limits<double>::quiet_NaN(), 1.001, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE(ok);
}

TEST_CASE("ECI norm beyond tolerance fails", "[qgoal][validation][boundary]") {
    MatX q_goal(4, 1);
    q_goal.col(0) << std::numeric_limits<double>::quiet_NaN(), 1.0011, 0.0, 0.0;
    std::string error_msg;

    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 0 ECI direction is not normalized");
}

TEST_CASE("Error pinpoints failing column in mixed set", "[qgoal][validation]") {
    MatX q_goal(4, 3);
    q_goal.col(0) << 1.0, 0.0, 0.0, 0.0;
    q_goal.col(1) << std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0, 0.0;
    q_goal.col(2) << 0.5, 0.5, 0.5, 0.5;
    q_goal(2, 2) = std::numeric_limits<double>::quiet_NaN();

    std::string error_msg;
    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "q_goal column 2 has invalid NaN/Inf in rows 2-4");
}
