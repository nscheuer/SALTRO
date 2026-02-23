#pragma once

#include <array>
#include <Eigen/Dense>

template<int R, int C, int D>
class Tensor3 {
public:
    using Slice = Eigen::Matrix<double, R, C>;

    Tensor3() { setZero(); }

    static Tensor3 Zero() { return Tensor3(); }

    void setZero() {
        for (auto& s : data_) s.setZero();
    }

    Slice& slice(int k) { return data_[static_cast<std::size_t>(k)]; }
    const Slice& slice(int k) const { return data_[static_cast<std::size_t>(k)]; }

    double& operator()(int i, int j, int k) {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

    const double& operator()(int i, int j, int k) const {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

private:
    std::array<Slice, D> data_{};
};
    

class Actuator {
public:
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using BaseState = Eigen::Matrix<double, 7, 1>;

    using Mat13 = Eigen::Matrix<double, 1, 3>;
    using Mat73 = Eigen::Matrix<double, 7, 3>;

    using T113 = Tensor3<1, 1, 3>;
    using T173 = Tensor3<1, 7, 3>;
    using T773 = Tensor3<7, 7, 3>;

    static constexpr int input_len = 1;

    Actuator() = delete;
    Actuator(const Vec3 &axis, double u_max);
    virtual ~Actuator() = default;

    const Vec3& axis() const noexcept;
    double u_max() const noexcept;

    double clamp(double u) const noexcept;

    virtual Vec3 torque(double u, const BaseState& x) const;

    virtual Mat13 dtorq_du(double u, const BaseState& x) const;
    virtual Mat73 dtorq_dbasestate(double u, const BaseState& x) const;

    virtual T113 ddtorq_dudu(double u, const BaseState& x) const;
    virtual T173 ddtorq_dudbasestate(double u, const BaseState& x) const;
    virtual T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x) const;

protected:
    static Vec3 normalize(const Vec3& v);

    Vec3 axis_;
    double u_max_;
};