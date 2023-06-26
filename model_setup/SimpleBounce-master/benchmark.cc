// benchmark point for mS = 5 GeV and sin\theta = 0.17
#define _USE_MATH_DEFINES // for C++
#include <iostream>
#include <cmath>
#include "simplebounce.h"
#include <sys/time.h>
#include <numbers>
using namespace std;
using namespace simplebounce;

class MyModel : public GenericModel
{
public:
    double lambda;
    double muHsq;
    double muSsq;
    double A;
    double g2;
    double g1;
    double yt;
    double Qsq;
    double v;
    double f;
    double delta;
    MyModel()
    {
        setNphi(2);
        lambda = 0.1491090009193912;
        A = 0.006642284413988053;
        muHsq = 0.008755523620463991;
        muSsq = 0.00018132516900000002;
        g1 = 0.3573944734603928;
        g2 = 0.6510161183564389;
        yt = 0.9777726923522626;
        Qsq = 0.0911876 * 0.0911876;
        f = 3.680;
        v = 0.24622;
        delta = M_PI * 2 / 5;
    }

    // potential of scalar field(s)
    double vpot(const double *phi) const
    {
        double vtree = (0.25 * lambda * phi[0] * phi[0] * phi[0] * phi[0] - 0.5 * (muHsq - A * f * cos(delta)) * phi[0] * phi[0] + muSsq * f * f * (1 - cos(phi[1] / f)) - 0.5 * A * f * cos(phi[1] / f - delta) * (phi[0] * phi[0] - v * v));
        double mW = (g2 * g2 * phi[0] * phi[0] * 0.25);
        double mZ = ((g2 * g2 + g1 * g1) * phi[0] * phi[0] * 0.25);
        double mt = (yt * yt * phi[0] * phi[0] * 0.5);
        double vcw = (6 * mW * mW * (log(mW / Qsq) - 5. / 6) + 3 * mZ * mZ * (log(mZ / Qsq) - 5. / 6) - 12. * mt * mt * (log(mt / Qsq) - 1.5)) / (64 * M_PI * M_PI);
        return vtree + vcw;
    }

    // derivative of potential of scalar field(s)
    void calcDvdphi(const double *phi, double *dvdphi) const
    {
        double mW = (g2 * g2 * phi[0] * phi[0] * 0.25);
        double mZ = ((g2 * g2 + g1 * g1) * phi[0] * phi[0] * 0.25);
        double mt = (yt * yt * phi[0] * phi[0] * 0.5);
        double mWd = (g2 * g2 * phi[0] * 0.5);
        double mZd = ((g2 * g2 + g1 * g1) * phi[0] * 0.5);
        double mtd = (yt * yt * phi[0]);
        dvdphi[0] = lambda * phi[0] * phi[0] * phi[0] - (muHsq - A * f * cos(delta)) * phi[0] - A * f * cos(phi[1] / f - delta) * phi[0] + (6 * mW * mWd * (2 * log(mW / Qsq) - 2. / 3) + 3 * mZ * mZd * (2 * log(mZ / Qsq) - 2. / 3) - 12 * mt * mtd * (2 * log(mt / Qsq) - 2.)) / (64 * M_PI * M_PI);
        dvdphi[1] = muSsq * f * sin(phi[1] / f) + 0.5 * A * (phi[0] * phi[0] - v * v) * sin(phi[1] / f - delta);
    }

    //    void dvdphi(const double* phi) const{
    //      double dv[2];
    //      double mW = (g*g*phi[0]*phi[0]*0.25);
    //      double mZ = ((g*g+gY*gY)*phi[0]*phi[0]*0.25);
    //      double mt = (yt*yt*phi[0]*phi[0]*0.5);
    //      double mWd = (g*g*phi[0]*0.5);
    //      double mZd = ((g*g+gY*gY)*phi[0]*0.5);
    //      double mtd = (yt*yt*phi[0]);
    //      dv[0] = lambda*phi[0]*phi[0]*phi[0] - muH*muH*phi[0]-A*phi[1]*phi[0]+(6*mW*mWd*(2*log(mW/Qsq)-2./3)+3*mZ*mZd*(2*log(mZ/Qsq)-2./3)-12*mt*mtd*(2*log(mt/Qsq)-2.))/(64*M_PI*M_PI);
    //      dv[1] = muS*muS*phi[1]-0.5*A*phi[0]*phi[0];
    //      cout << "dvdh = " << dv[0] << ", dvdS = " << dv[1] << endl;
    //    }
};

int main()
{

    BounceCalculator bounce;
    bounce.verboseOn();
    bounce.setRmax(1);      // phi(rmax) = phi(False vacuum)
    bounce.setDimension(4); // number of space dimension
    bounce.setN(100);       // number of grid
    MyModel model;
    // double location[2] = {2.46.,0.};
    // cout << model.vpot(location) << endl;
    // model.dvdphi(location);
    bounce.setModel(&model);

    double phiTV[2] = {250.0, 4.63}; // a point at which V<0
    double phiFV[2] = {.24622, 0.};  // false vacuum
    bounce.setVacuum(phiTV, phiFV);
    cout << "potential at the minimum: " << model.vpot(phiFV) << endl;
    // calcualte the bounce solution
    bounce.solve();

    bounce.printBounce();
    // Euclidean action
    cout << "S_E = " << bounce.action() << endl;

    return 0;
}