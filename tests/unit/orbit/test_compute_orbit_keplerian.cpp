#include <iostream>
#include <Eigen/Dense>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>
#include <saltro/limits.h>

int main() {
    Eigen::Vector3d r0(7000e3, 0, 0);
    Eigen::Vector3d v0(0, 7500, 0);

    Eigen::Matrix<double,1,saltro::limits::MAX_LENGTH_TRAJ> t;
    for(int i=0;i<100;i++)
        t(i) = i * 1e-5; 

    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,saltro::limits::MAX_LENGTH_TRAJ> V;

    bool ok = saltro::orbit::compute_orbit_keplerian(r0,v0,t,100,R,V);

     if(!ok){
        std::cout<<"FAIL\n";
        return 1;
    }

    std::cout<<"R0: "<<R.col(0).transpose()<<"\n";
    std::cout<<"R10: "<<R.col(10).transpose()<<"\n";

    return 0;
}