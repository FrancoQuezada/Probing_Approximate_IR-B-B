#ifndef IndINCLUDE
#define IndINCLUDE
#include "Header.h"
#endif

double Approx_Branch_and_Bound(IloEnv env, IloNumArray Solution, int bbMode, int evalMode, int bbCache) {
    const int prevSetupCertEnable = SetupCertEnable;
    SetupCertEnable = 0;

    cout << "AIR-B&B module" << endl;
    cout << "  eps=" << ApproxGapEps
         << " reuseMode=" << ApproxReuseMode
         << " localImproveMode=" << ApproxLocalImproveMode
         << " lbMode=" << ApproxLBMode
         << " globalLagLB=" << ApproxGlobalLagLBMode
         << " exactFallback=" << ApproxExactFallbackMode
         << endl;

    double out = Branch_and_Bound_Ma(env, Solution, bbMode, evalMode, bbCache);
    SetupCertEnable = prevSetupCertEnable;
    return out;
}
