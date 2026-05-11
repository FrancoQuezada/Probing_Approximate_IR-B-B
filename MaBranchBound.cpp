#ifndef IndINCLUDE
#define IndINCLUDE
#include "Header.h"
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <numeric>
#endif

struct BBNode {
    int s1_mask;
    int s0_mask;
    std::vector<int> blocks;
    double lb;
    double ub;
    int depth;
};

struct RLObjective {
    double lb;
    double ub;
};

struct TauReuseCandidate {
    std::vector<std::vector<int>> y;
    double obj = IloInfinity;
    bool optimal = false;
    bool hasY = false;
    double certLB = -IloInfinity;
    bool hasCertLB = false;
    bool certExact = false;
};

struct TauReuseEntryDyn {
    std::vector<uint64_t> bits;
    int card = 0;
    std::vector<std::vector<int>> y;
    double obj = IloInfinity;
    bool optimal = false;
    bool hasY = false;
    double certLB = -IloInfinity;
    bool hasCertLB = false;
    bool certExact = false;
};

struct TauReuseRankedCandidate {
    std::vector<std::vector<int>> y;
    uint64_t mask64 = 0ULL;
    std::vector<uint64_t> bits;
    int card = 0;
    double coverage = 0.0;
    double normObj = IloInfinity;
    bool optimal = false;
};

struct TauRelaxedLPInfo {
    double lpLB = IloInfinity;
    double lpStrongLB = IloInfinity;
    std::vector<double> lambda;
    bool valid = false;
    bool strongValid = false;
};

struct IntVecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

static std::unordered_map<std::string, std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>> GlobalPhiCache;
static std::vector<std::vector<std::vector<int>>> TauPbLists;
static std::vector<std::vector<std::string>> TauKeys;
static std::vector<char> TauCacheBuilt;
static std::unordered_map<uint64_t, TauReuseCandidate> TauReuseYCache;
static std::vector<uint64_t> TauReuseYMasks;
static std::unordered_map<std::string, size_t> TauReuseDynIndexByKey;
static std::vector<TauReuseEntryDyn> TauReuseDynEntries;
static std::unordered_map<std::string, double> TauApproxGlobalLBCache;
static std::unordered_map<std::string, TauRelaxedLPInfo> TauRelaxedLPCache;
static std::unordered_map<std::string, double> TauCertCompLagCache;
static std::unordered_map<std::string, double> TauFinalLBCache;
static bool BBTimeControlActive = false;
static bool BBTimeControlHit = false;
static std::chrono::steady_clock::time_point BBTimeDeadline;

static inline void StartBBTimeControl(double seconds) {
    BBTimeControlActive = true;
    BBTimeControlHit = false;
    BBTimeDeadline = std::chrono::steady_clock::now() +
                     std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(std::max(0.0, seconds)));
}

static inline void StopBBTimeControl() {
    BBTimeControlActive = false;
    BBTimeControlHit = false;
}

static inline bool BBTimeLimitReached() {
    if (!BBTimeControlActive) {
        return false;
    }
    if (BBTimeControlHit) {
        return true;
    }
    if (std::chrono::steady_clock::now() >= BBTimeDeadline) {
        BBTimeControlHit = true;
        return true;
    }
    return false;
}

static inline bool ApproxGapSatisfiedWithLb(double lb, double ub, double epsTol) {
    if (!std::isfinite(lb) || !std::isfinite(ub)) {
        return false;
    }
    const double denom = std::max(1.0, std::fabs(lb));
    return ((ub - lb) / denom) <= epsTol + 1e-9;
}

class StopOnApproxGapIncumbentI : public IloCplex::MIPInfoCallbackI {
    double externalLb_;
    double epsTol_;
public:
    StopOnApproxGapIncumbentI(IloEnv env, double externalLb, double epsTol)
        : IloCplex::MIPInfoCallbackI(env), externalLb_(externalLb), epsTol_(epsTol) {}

    void main() ILO_OVERRIDE {
        if (!hasIncumbent()) {
            return;
        }
        const double inc = getIncumbentObjValue();
        double lbDyn = externalLb_;
        double bestBound = getBestObjValue();
        if (std::isfinite(bestBound) && (!std::isfinite(lbDyn) || bestBound > lbDyn)) {
            lbDyn = bestBound;
        }
        if (ApproxGapSatisfiedWithLb(lbDyn, inc, epsTol_)) {
            abort();
        }
    }

    IloCplex::CallbackI* duplicateCallback() const ILO_OVERRIDE {
        return new (getEnv()) StopOnApproxGapIncumbentI(getEnv(), externalLb_, epsTol_);
    }
};

static std::string BuildTauKeyFromPbList(const std::vector<int>& pbList) {
    std::string key;
    key.reserve(pbList.size() * 6);
    for (int pb : pbList) {
        key.append(std::to_string(pb));
        key.push_back(',');
    }
    return key;
}

static size_t FixPairIndex(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    size_t base = static_cast<size_t>(i) * (2 * n - i - 1) / 2;
    return base + static_cast<size_t>(j - i - 1);
}

static void BuildFixingRsets(std::vector<std::vector<uint64_t>>& R) {
    const int n = N_pb;
    const size_t m = static_cast<size_t>(n) * static_cast<size_t>(n - 1) / 2;
    const size_t blocks = (m + 63) / 64;
    R.assign(N_pn, std::vector<uint64_t>(blocks, 0ULL));
    for (int j = 0; j < N_pn; ++j) {
        for (int pb = 0; pb < N_pb; ++pb) {
            for (int pb2 = pb + 1; pb2 < N_pb; ++pb2) {
                long long q1 = llround(eta[j][pb] * EtaRoundingScale);
                long long q2 = llround(eta[j][pb2] * EtaRoundingScale);
                if (q1 != q2) {
                    size_t idx = FixPairIndex(pb, pb2, N_pb);
                    R[j][idx >> 6] |= (1ULL << (idx & 63));
                }
            }
        }
    }
}

static inline bool FixSubset(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~b[i]) != 0ULL) {
            return false;
        }
    }
    return true;
}

static inline bool FixSubsetUnion2(const std::vector<uint64_t>& a,
                                   const std::vector<uint64_t>& u) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~u[i]) != 0ULL) {
            return false;
        }
    }
    return true;
}

static inline bool FixSubsetUnion2Plus(const std::vector<uint64_t>& a,
                                       const std::vector<uint64_t>& u,
                                       const std::vector<uint64_t>& r) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~(u[i] | r[i])) != 0ULL) {
            return false;
        }
    }
    return true;
}

static inline bool FixSubsetUnion2Plus2(const std::vector<uint64_t>& a,
                                        const std::vector<uint64_t>& u,
                                        const std::vector<uint64_t>& r1,
                                        const std::vector<uint64_t>& r2) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~(u[i] | r1[i] | r2[i])) != 0ULL) {
            return false;
        }
    }
    return true;
}

static inline bool FixIsEmpty(const std::vector<uint64_t>& a) {
    for (uint64_t x : a) {
        if (x != 0ULL) {
            return false;
        }
    }
    return true;
}

static int ApplyResidualRNodeFixing(
    const std::vector<std::vector<uint64_t>>& baseR,
    const std::vector<int>& idxByAlpha,
    int s1_mask,
    int s0_mask,
    int& addMask
) {
    addMask = 0;
    if (N_pb < 2 || N_pn <= 0 || baseR.empty()) {
        return 0;
    }
    const int allMask = (N_pn >= 31) ? -1 : ((1 << N_pn) - 1);
    const int freeMask = allMask & ~(s0_mask | s1_mask);
    if (freeMask == 0) {
        return 0;
    }

    const size_t blocks = baseR[0].size();
    std::vector<uint64_t> rS1(blocks, 0ULL);
    for (int j = 0; j < N_pn; ++j) {
        if (((s1_mask >> j) & 1) == 0) continue;
        for (size_t b = 0; b < blocks; ++b) {
            rS1[b] |= baseR[j][b];
        }
    }

    std::vector<std::vector<uint64_t>> residualLhs(N_pn, std::vector<uint64_t>(blocks, 0ULL));
    std::vector<unsigned char> fixed(N_pn, 1);
    for (int j = 0; j < N_pn; ++j) {
        if (((freeMask >> j) & 1) == 0) continue;
        fixed[j] = 0;
        for (size_t b = 0; b < blocks; ++b) {
            residualLhs[j][b] = baseR[j][b] & ~rS1[b];
        }
    }

    const size_t pnPairs = static_cast<size_t>(N_pn) * static_cast<size_t>(N_pn - 1) / 2;
    std::vector<std::vector<uint64_t>> union2(pnPairs, std::vector<uint64_t>(blocks, 0ULL));
    for (int i = 0; i < N_pn; ++i) {
        if (((freeMask >> i) & 1) == 0) continue;
        for (int j = i + 1; j < N_pn; ++j) {
            if (((freeMask >> j) & 1) == 0) continue;
            size_t p = FixPairIndex(i, j, N_pn);
            for (size_t b = 0; b < blocks; ++b) {
                union2[p][b] = baseR[i][b] | baseR[j][b];
            }
        }
    }

    std::vector<int> fix3, fix4, fix5;

    // Proposition 3 (node residual sets)
    for (int k = 0; k < N_pn; ++k) {
        if (fixed[k]) continue;
        if (FixIsEmpty(residualLhs[k])) {
            fix3.push_back(k);
            fixed[k] = 1;
            continue;
        }
        for (int idx : idxByAlpha) {
            if (((freeMask >> idx) & 1) == 0) continue;
            if (idx == k) continue;
            if (alpha[idx] > alpha[k]) break;
            if (FixSubset(residualLhs[k], baseR[idx])) {
                fix3.push_back(k);
                fixed[k] = 1;
                break;
            }
        }
    }

    // Proposition 4 (node residual sets)
    for (int k = 0; k < N_pn; ++k) {
        if (fixed[k]) continue;
        bool fixedLocal = false;
        std::vector<int> candidates;
        candidates.reserve(N_pn);
        for (int idx : idxByAlpha) {
            if (((freeMask >> idx) & 1) == 0) continue;
            if (idx == k) continue;
            if (fixed[idx]) continue;
            if (alpha[idx] > alpha[k]) break;
            candidates.push_back(idx);
        }
        const int csz = static_cast<int>(candidates.size());
        for (int a = 0; a < csz && !fixedLocal; ++a) {
            int i = candidates[a];
            if (alpha[i] > alpha[k]) break;
            for (int b = a + 1; b < csz && !fixedLocal; ++b) {
                int j = candidates[b];
                if (alpha[i] + alpha[j] > alpha[k]) break;
                size_t p = FixPairIndex(i, j, N_pn);
                if (FixSubsetUnion2(residualLhs[k], union2[p])) {
                    fix4.push_back(k);
                    fixedLocal = true;
                }
            }
        }
        if (fixedLocal) {
            fixed[k] = 1;
        }
    }

    // Proposition 5 (node residual sets, max union size = 4)
    for (int k = 0; k < N_pn; ++k) {
        if (fixed[k]) continue;
        bool fixedLocal = false;
        std::vector<int> candidates;
        candidates.reserve(N_pn);
        for (int idx : idxByAlpha) {
            if (((freeMask >> idx) & 1) == 0) continue;
            if (idx == k) continue;
            if (fixed[idx]) continue;
            if (alpha[idx] > alpha[k]) break;
            candidates.push_back(idx);
        }
        const int csz = static_cast<int>(candidates.size());
        for (int a = 0; a < csz && !fixedLocal; ++a) {
            int i = candidates[a];
            if (alpha[i] > alpha[k]) break;
            for (int b = a + 1; b < csz && !fixedLocal; ++b) {
                int j = candidates[b];
                if (alpha[i] + alpha[j] > alpha[k]) break;
                size_t p = FixPairIndex(i, j, N_pn);
                for (int c = b + 1; c < csz && !fixedLocal; ++c) {
                    int l = candidates[c];
                    if (alpha[i] + alpha[j] + alpha[l] > alpha[k]) break;
                    if (FixSubsetUnion2Plus(residualLhs[k], union2[p], baseR[l])) {
                        fix5.push_back(k);
                        fixedLocal = true;
                        break;
                    }
                    for (int d = c + 1; d < csz && !fixedLocal; ++d) {
                        int m = candidates[d];
                        if (alpha[i] + alpha[j] + alpha[l] + alpha[m] > alpha[k]) break;
                        if (FixSubsetUnion2Plus2(residualLhs[k], union2[p], baseR[l], baseR[m])) {
                            fix5.push_back(k);
                            fixedLocal = true;
                            break;
                        }
                    }
                }
            }
        }
        if (fixedLocal) {
            fixed[k] = 1;
        }
    }

    for (int k : fix3) addMask |= (1 << k);
    for (int k : fix4) addMask |= (1 << k);
    for (int k : fix5) addMask |= (1 << k);
    addMask &= freeMask;
    return __builtin_popcount(static_cast<unsigned int>(addMask));
}

static void BuildPbListForTau(int zMask, int tauRep, std::vector<int>& pbList);

struct CapBBNode {
    std::vector<int> L;
    std::vector<int> U;
    double lb;
    double ub;
};

static double MinLBStack(const std::vector<CapBBNode>& stack) {
    double minLb = IloInfinity;
    for (const auto& node : stack) {
        if (node.lb < minLb) {
            minLb = node.lb;
        }
    }
    return minLb;
}

static inline int CapIdx(int it, int t) {
    return it * (N_tp + 1) + t;
}

static double PhiItemCapCached(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<int>& kVec,
    std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>& cache
);

static bool CapBoundsFeasibleEq(const std::vector<int>& L, const std::vector<int>& U);

static void EnsureTauCache(int zMask) {
    if ((int)TauCacheBuilt.size() != N_z) {
        TauPbLists.assign(N_z, {});
        TauKeys.assign(N_z, {});
        TauCacheBuilt.assign(N_z, 0);
    }
    if (TauCacheBuilt[zMask]) {
        return;
    }
    const std::vector<int>& tauReps = TauReps[zMask];
    TauPbLists[zMask].resize(tauReps.size());
    TauKeys[zMask].resize(tauReps.size());
    for (size_t i = 0; i < tauReps.size(); ++i) {
        BuildPbListForTau(zMask, tauReps[i], TauPbLists[zMask][i]);
        std::string key;
        const auto& pbList = TauPbLists[zMask][i];
        key.reserve(pbList.size() * 6);
        for (int pb : pbList) {
            key.append(std::to_string(pb));
            key.push_back(',');
        }
        TauKeys[zMask][i] = std::move(key);
    }
    TauCacheBuilt[zMask] = 1;
}

static void TightenU_ByDemand(
    const std::vector<int>& pbList,
    std::vector<int>& U
){
    if (pbList.empty()) {
        return;
    }
    std::vector<std::vector<int>> maxRemain(N_it, std::vector<int>(N_tp + 1, 0));
    for (int it = 0; it < N_it; ++it) {
        for (int pb : pbList) {
            for (int sc = 0; sc < N_sc; ++sc) {
                double cum = 0.0;
                for (int t = N_tp; t >= 1; --t) {
                    cum += d[t][sc][pb][it];
                    int val = static_cast<int>(std::ceil(cum));
                    if (val > maxRemain[it][t]) {
                        maxRemain[it][t] = val;
                    }
                }
            }
        }
    }
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int idx = CapIdx(it, t);
            U[idx] = std::min(U[idx], maxRemain[it][t]);
        }
    }
}

static bool PropagateU_ByResidual(std::vector<int>& L, std::vector<int>& U) {
    for (int t = 1; t <= N_tp; ++t) {
        int sumL = 0;
        for (int it = 0; it < N_it; ++it) {
            sumL += L[CapIdx(it, t)];
        }
        if (sumL > Cap) {
            return false;
        }
        int residual = Cap - sumL;
        for (int it = 0; it < N_it; ++it) {
            int idx = CapIdx(it, t);
            U[idx] = std::min(U[idx], L[idx] + residual);
            if (L[idx] > U[idx]) {
                return false;
            }
        }
    }
    return true;
}

static bool PropagateL_BySumU(std::vector<int>& L, std::vector<int>& U) {
    for (int t = 1; t <= N_tp; ++t) {
        int sumU = 0;
        for (int it = 0; it < N_it; ++it) {
            sumU += U[CapIdx(it, t)];
        }
        if (sumU < Cap) {
            return false;
        }
        for (int it = 0; it < N_it; ++it) {
            int idx = CapIdx(it, t);
            int minLi = Cap - (sumU - U[idx]);
            if (minLi > L[idx]) {
                L[idx] = minLi;
                if (L[idx] > U[idx]) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool SumU_EqCapAll(const std::vector<int>& U) {
    for (int t = 1; t <= N_tp; ++t) {
        int sumU = 0;
        for (int it = 0; it < N_it; ++it) {
            sumU += U[CapIdx(it, t)];
        }
        if (sumU != Cap) {
            return false;
        }
    }
    return true;
}

static bool SumL_EqCapAll(const std::vector<int>& L) {
    for (int t = 1; t <= N_tp; ++t) {
        int sumL = 0;
        for (int it = 0; it < N_it; ++it) {
            sumL += L[CapIdx(it, t)];
        }
        if (sumL != Cap) {
            return false;
        }
    }
    return true;
}

static bool NormalizeNodeBounds(std::vector<int>& L, std::vector<int>& U) {
    if (!CapBoundsFeasibleEq(L, U)) {
        return false;
    }
    if (!PropagateU_ByResidual(L, U)) {
        return false;
    }
    if (!PropagateL_BySumU(L, U)) {
        return false;
    }
    if (!CapBoundsFeasibleEq(L, U)) {
        return false;
    }
    return true;
}

static double PhiSumAtU(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<int>& U,
    std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>& phiCache
){
    double total = 0.0;
    std::vector<int> kVec(N_tp + 1, 0);
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            kVec[t] = U[CapIdx(it, t)];
        }
        total += PhiItemCapCached(env, it, pbList, kVec, phiCache);
    }
    return total;
}

static int CountBits(int mask) {
    int cnt = 0;
    while (mask) {
        cnt += (mask & 1);
        mask >>= 1;
    }
    return cnt;
}

static std::string MaskToBits(int mask) {
    std::string out;
    out.reserve(N_pn * 2);
    for (int i = 0; i < N_pn; ++i) {
        out.push_back(((mask >> i) & 1) ? '1' : '0');
        if (i < N_pn - 1) {
            out.push_back(',');
        }
    }
    return out;
}

static double AlphaMask(int mask) {
    double val = 0.0;
    for (int i = 0; i < N_pn; ++i) {
        if ((mask >> i) & 1) {
            val += alpha[i];
        }
    }
    return val;
}

// Collect pb indices for a tau class under a probing vector.
static void BuildPbListForTau(int zMask, int tauRep, std::vector<int>& pbList) {
    pbList.clear();
    for (int pb = 0; pb < N_pb; ++pb) {
        if (KappaTau[zMask][pb] == tauRep) {
            pbList.push_back(pb);
        }
    }
}

static double SolveValueBaseItem_BB(IloEnv env, int item) {
    const IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp + 1, 0, 1, ILOBOOL);
    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, N_pb * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, N_pb * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int pb = 0; pb < N_pb; ++pb) {
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = pb * (N_tp + 1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        model.add(X[t] <= std::min(Cap, (int)BigM_np[item][t]) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int sc = 0; sc < N_sc; ++sc) {
        for (int pb = 0; pb < N_pb; ++pb) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = pb * (N_tp + 1) + t;
                int idxPrev = pb * (N_tp + 1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = pb * (N_tp + 1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }
    if (cplex.getStatus() != IloAlgorithm::Optimal) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double val = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return val;
}

static double SolveValueTauItem_BB(IloEnv env, int item, const std::vector<int>& pbList) {
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    const IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp + 1, 0, 1, ILOBOOL);
    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        model.add(X[t] <= std::min(Cap, (int)BigM_np[item][t]) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                int idxPrev = k * (N_tp + 1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp + 1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }
    if (cplex.getStatus() != IloAlgorithm::Optimal) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double val = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return val;
}

static void ComputeValueFunctionBounds_BB(
    IloEnv env,
    std::vector<double>& vBase,
    std::vector<std::vector<double>>& vComp
){
    vBase.assign(N_it, 0.0);
    vComp.assign(N_it, std::vector<double>(N_pn, 0.0));
    for (int it = 0; it < N_it; ++it) {
        std::unordered_map<std::string, double> tauCache;
        vBase[it] = SolveValueBaseItem_BB(env, it);
        for (int j = 0; j < N_pn; ++j) {
            const int zMask = 1 << j;
            double total = 0.0;
            const std::vector<int>& tauReps = TauReps[zMask];
            for (int tauRep : tauReps) {
                std::vector<int> pbList;
                BuildPbListForTau(zMask, tauRep, pbList);
                if (pbList.empty()) continue;
                std::string key;
                key.reserve(pbList.size() * 6);
                for (int pb : pbList) {
                    key.append(std::to_string(pb));
                    key.push_back(',');
                }
                auto itCache = tauCache.find(key);
                if (itCache != tauCache.end()) {
                    total += itCache->second;
                } else {
                    double val = SolveValueTauItem_BB(env, it, pbList);
                    tauCache[key] = val;
                    total += val;
                }
            }
            vComp[it][j] = total;
        }
    }
}

static std::vector<double> VF_Base_BB;
static std::vector<std::vector<double>> VF_Comp_BB;
static bool VF_Ready_BB = false;
static bool VF_Available_BB = false;

static bool UseVFInSubproblems_BB() {
    return (VI_pro3 == 2 || VI_pro3 == 3);
}

static void PrepareValueFunction_BB(IloEnv env) {
    if (VF_Ready_BB) {
        return;
    }
    ComputeValueFunctionBounds_BB(env, VF_Base_BB, VF_Comp_BB);
    VF_Available_BB = true;
    if (static_cast<int>(VF_Base_BB.size()) != N_it || static_cast<int>(VF_Comp_BB.size()) != N_it) {
        VF_Available_BB = false;
    }
    for (int it = 0; VF_Available_BB && it < N_it; ++it) {
        if (!std::isfinite(VF_Base_BB[it])) {
            VF_Available_BB = false;
            break;
        }
        if (static_cast<int>(VF_Comp_BB[it].size()) != N_pn) {
            VF_Available_BB = false;
            break;
        }
        for (int j = 0; j < N_pn; ++j) {
            if (!std::isfinite(VF_Comp_BB[it][j])) {
                VF_Available_BB = false;
                break;
            }
        }
    }
    if (!VF_Available_BB) {
        cout << "VF bounds disabled in wht 24: auxiliary VF solves not optimal/finite." << endl;
    }
    VF_Ready_BB = true;
}

static double ValueFunctionRHS_BB(int item, int zMask) {
    double rhs = VF_Base_BB[item];
    for (int j = 0; j < N_pn; ++j) {
        double vj = VF_Comp_BB[item][j];
        if (!std::isfinite(vj) || vj > VF_Base_BB[item]) {
            vj = VF_Base_BB[item];
        }
        double coeff = VF_Base_BB[item] - vj;
        if (coeff > 0.0 && ((zMask >> j) & 1)) {
            rhs -= coeff;
        }
    }
    return rhs;
}

static double ValueFunctionLB_BB(int zMask) {
    double total = 0.0;
    for (int it = 0; it < N_it; ++it) {
        total += ValueFunctionRHS_BB(it, zMask);
    }
    return total;
}

static double ApplyValueFunctionLB_BB(IloEnv env, int zMask, double total) {
    if (!UseVFInSubproblems_BB()) {
        return total;
    }
    PrepareValueFunction_BB(env);
    if (!VF_Available_BB) {
        return total;
    }
    double vfLb = ValueFunctionLB_BB(zMask);
    if (std::isfinite(vfLb) && vfLb > total) {
        return vfLb;
    }
    return total;
}

struct EvalWithOpt {
    double val;
    bool optimal;
};

struct EvalStats {
    int tauTotal = 0;
    int tauSolved = 0;
    int itemSolved = 0;
    int jointSolved = 0;
    double timeItem = 0.0;
    double timeJoint = 0.0;
};

struct CertAggStats {
    long long finiteCount = 0;
    double ubSum = 0.0;
    double lbSum = 0.0;
};

struct AirTauEvalResult {
    double val = IloInfinity;
    double lb = -IloInfinity;
    bool exact = false;
    bool hasY = false;
    std::vector<std::vector<int>> y;
};

struct TauHeuristicSeed {
    bool valid = false;
    std::vector<std::vector<int>> y;
    std::vector<std::vector<double>> x;
    std::vector<double> itemObjLB;
    std::vector<double> sumX;
    double actualObj = IloInfinity;
    double lagObj = IloInfinity;
};

struct TauLagResult {
    double lb = -IloInfinity;
    double ub = IloInfinity;
    TauHeuristicSeed finalSeed;
    bool hasBestUbY = false;
    std::vector<std::vector<int>> bestUbY;
};

struct TauSubsetCacheEntry {
    double bestAny = 0.0;
    double bestOpt = 0.0;
    double bestLb = 0.0;
    bool hasAny = false;
    bool hasOpt = false;
    bool hasLb = false;
};

struct TauPackingMemoEntry {
    long long epochAny = -1;
    long long epochOpt = -1;
    long long epochLb = -1;
    double lbAny = 0.0;
    double lbOpt = 0.0;
    double lbFromLb = 0.0;
};

static bool PbListToMask64(const std::vector<int>& pbList, uint64_t& outMask) {
    outMask = 0ULL;
    if (N_pb <= 0 || N_pb > 63) {
        return false;
    }
    for (int pb : pbList) {
        if (pb < 0 || pb >= 63) {
            return false;
        }
        outMask |= (1ULL << static_cast<unsigned int>(pb));
    }
    return true;
}

static void Mask64ToPbList(uint64_t mask, std::vector<int>& pbList) {
    pbList.clear();
    while (mask) {
        uint64_t bit = mask & (~mask + 1ULL);
        int pb = __builtin_ctzll(bit);
        pbList.push_back(pb);
        mask ^= bit;
    }
}

static void RegisterTauSubsetValue(
    uint64_t tauMask,
    double val,
    bool optimal,
    std::unordered_map<uint64_t, TauSubsetCacheEntry>& subsetCache,
    std::vector<uint64_t>& subsetMasks,
    long long& subsetEpoch
) {
    if (tauMask == 0ULL || !std::isfinite(val)) {
        return;
    }
    auto it = subsetCache.find(tauMask);
    if (it == subsetCache.end()) {
        TauSubsetCacheEntry entry;
        it = subsetCache.emplace(tauMask, entry).first;
        subsetMasks.push_back(tauMask);
    }
    TauSubsetCacheEntry& entry = it->second;
    bool updated = false;
    if (!entry.hasAny || val > entry.bestAny + 1e-9) {
        entry.bestAny = val;
        entry.hasAny = true;
        updated = true;
    }
    if (optimal && (!entry.hasOpt || val > entry.bestOpt + 1e-9)) {
        entry.bestOpt = val;
        entry.hasOpt = true;
        updated = true;
    }
    if (updated) {
        subsetEpoch++;
    }
}

static void RegisterTauSubsetLowerBound(
    uint64_t tauMask,
    double lb,
    std::unordered_map<uint64_t, TauSubsetCacheEntry>& subsetCache,
    std::vector<uint64_t>& subsetMasks,
    long long& subsetEpoch
) {
    if (tauMask == 0ULL || !std::isfinite(lb)) {
        return;
    }
    auto it = subsetCache.find(tauMask);
    if (it == subsetCache.end()) {
        TauSubsetCacheEntry entry;
        it = subsetCache.emplace(tauMask, entry).first;
        subsetMasks.push_back(tauMask);
    }
    TauSubsetCacheEntry& entry = it->second;
    if (!entry.hasLb || lb > entry.bestLb + 1e-9) {
        entry.bestLb = lb;
        entry.hasLb = true;
        subsetEpoch++;
    }
}

static double ComputeTauPackingLb(
    uint64_t targetMask,
    bool allowHeuristicValues,
    bool excludeExactTarget,
    const std::unordered_map<uint64_t, TauSubsetCacheEntry>& subsetCache,
    const std::vector<uint64_t>& subsetMasks
) {
    if (targetMask == 0ULL) {
        return 0.0;
    }
    std::vector<std::pair<uint64_t, double>> candidates;
    candidates.reserve(subsetMasks.size());
    for (uint64_t mask : subsetMasks) {
        if (mask == 0ULL || (mask & ~targetMask) != 0ULL) {
            continue;
        }
        if (excludeExactTarget && mask == targetMask) {
            continue;
        }
        auto it = subsetCache.find(mask);
        if (it == subsetCache.end()) {
            continue;
        }
        const TauSubsetCacheEntry& entry = it->second;
        double val = 0.0;
        bool ok = false;
        if (allowHeuristicValues) {
            if (entry.hasAny) {
                val = entry.bestAny;
                ok = true;
            }
        } else {
            if (entry.hasOpt) {
                val = entry.bestOpt;
                ok = true;
            }
        }
        if (!ok || !std::isfinite(val) || val <= 1e-12) {
            continue;
        }
        candidates.push_back({mask, val});
    }
    if (candidates.empty()) {
        return 0.0;
    }

    const int targetCard = __builtin_popcountll(targetMask);
    const int kExactPackingMaxCard = 14;
    if (targetCard <= kExactPackingMaxCard) {
        std::unordered_map<uint64_t, double> bestByLocalMask;
        bestByLocalMask.reserve(candidates.size());
        for (const auto& cand : candidates) {
            uint64_t localMask = 0ULL;
            uint64_t tmpTarget = targetMask;
            int bitPos = 0;
            while (tmpTarget) {
                uint64_t bit = tmpTarget & (~tmpTarget + 1ULL);
                if (cand.first & bit) {
                    localMask |= (1ULL << static_cast<unsigned int>(bitPos));
                }
                tmpTarget ^= bit;
                bitPos++;
            }
            auto it = bestByLocalMask.find(localMask);
            if (it == bestByLocalMask.end() || cand.second > it->second) {
                bestByLocalMask[localMask] = cand.second;
            }
        }

        const uint64_t stateCount = 1ULL << static_cast<unsigned int>(targetCard);
        std::vector<double> dp(stateCount, -IloInfinity);
        dp[0] = 0.0;
        for (const auto& kv : bestByLocalMask) {
            uint64_t cmask = kv.first;
            double cval = kv.second;
            for (long long s = static_cast<long long>(stateCount) - 1; s >= 0; --s) {
                if (!std::isfinite(dp[static_cast<size_t>(s)])) {
                    continue;
                }
                if ((static_cast<uint64_t>(s) & cmask) != 0ULL) {
                    continue;
                }
                uint64_t next = static_cast<uint64_t>(s) | cmask;
                double candVal = dp[static_cast<size_t>(s)] + cval;
                if (candVal > dp[static_cast<size_t>(next)]) {
                    dp[static_cast<size_t>(next)] = candVal;
                }
            }
        }
        double best = 0.0;
        for (double v : dp) {
            if (std::isfinite(v) && v > best) {
                best = v;
            }
        }
        return best;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<uint64_t, double>& a, const std::pair<uint64_t, double>& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return __builtin_popcountll(a.first) > __builtin_popcountll(b.first);
              });
    uint64_t used = 0ULL;
    double total = 0.0;
    for (const auto& cand : candidates) {
        if ((cand.first & used) != 0ULL) {
            continue;
        }
        used |= cand.first;
        total += cand.second;
    }
    return total;
}

static double GetTauPackingLb(
    uint64_t targetMask,
    bool allowHeuristicValues,
    bool excludeExactTarget,
    const std::unordered_map<uint64_t, TauSubsetCacheEntry>& subsetCache,
    const std::vector<uint64_t>& subsetMasks,
    std::unordered_map<uint64_t, TauPackingMemoEntry>& packingMemo,
    long long subsetEpoch
) {
    TauPackingMemoEntry& memo = packingMemo[targetMask];
    if (allowHeuristicValues) {
        if (memo.epochAny == subsetEpoch) {
            return memo.lbAny;
        }
        memo.lbAny = ComputeTauPackingLb(targetMask, true, excludeExactTarget, subsetCache, subsetMasks);
        memo.epochAny = subsetEpoch;
        return memo.lbAny;
    }
    if (memo.epochOpt == subsetEpoch) {
        return memo.lbOpt;
    }
    memo.lbOpt = ComputeTauPackingLb(targetMask, false, excludeExactTarget, subsetCache, subsetMasks);
    memo.epochOpt = subsetEpoch;
    return memo.lbOpt;
}

static double ComputeTauPackingLbFromLowerBounds(
    uint64_t targetMask,
    bool excludeExactTarget,
    const std::unordered_map<uint64_t, TauSubsetCacheEntry>& subsetCache,
    const std::vector<uint64_t>& subsetMasks
) {
    if (targetMask == 0ULL) {
        return 0.0;
    }
    std::vector<std::pair<uint64_t, double>> candidates;
    candidates.reserve(subsetMasks.size());
    for (uint64_t mask : subsetMasks) {
        if (mask == 0ULL || (mask & ~targetMask) != 0ULL) {
            continue;
        }
        if (excludeExactTarget && mask == targetMask) {
            continue;
        }
        auto it = subsetCache.find(mask);
        if (it == subsetCache.end()) {
            continue;
        }
        const TauSubsetCacheEntry& entry = it->second;
        if (!entry.hasLb || !std::isfinite(entry.bestLb) || entry.bestLb <= 1e-12) {
            continue;
        }
        candidates.push_back({mask, entry.bestLb});
    }
    if (candidates.empty()) {
        return 0.0;
    }

    const int targetCard = __builtin_popcountll(targetMask);
    const int kExactPackingMaxCard = 14;
    if (targetCard <= kExactPackingMaxCard) {
        std::unordered_map<uint64_t, double> bestByLocalMask;
        bestByLocalMask.reserve(candidates.size());
        for (const auto& cand : candidates) {
            uint64_t localMask = 0ULL;
            uint64_t tmpTarget = targetMask;
            int bitPos = 0;
            while (tmpTarget) {
                uint64_t bit = tmpTarget & (~tmpTarget + 1ULL);
                if (cand.first & bit) {
                    localMask |= (1ULL << static_cast<unsigned int>(bitPos));
                }
                tmpTarget ^= bit;
                bitPos++;
            }
            auto it = bestByLocalMask.find(localMask);
            if (it == bestByLocalMask.end() || cand.second > it->second) {
                bestByLocalMask[localMask] = cand.second;
            }
        }

        const uint64_t stateCount = 1ULL << static_cast<unsigned int>(targetCard);
        std::vector<double> dp(stateCount, -IloInfinity);
        dp[0] = 0.0;
        for (const auto& kv : bestByLocalMask) {
            uint64_t cmask = kv.first;
            double cval = kv.second;
            for (long long s = static_cast<long long>(stateCount) - 1; s >= 0; --s) {
                if (!std::isfinite(dp[static_cast<size_t>(s)])) {
                    continue;
                }
                if ((static_cast<uint64_t>(s) & cmask) != 0ULL) {
                    continue;
                }
                uint64_t next = static_cast<uint64_t>(s) | cmask;
                double candVal = dp[static_cast<size_t>(s)] + cval;
                if (candVal > dp[static_cast<size_t>(next)]) {
                    dp[static_cast<size_t>(next)] = candVal;
                }
            }
        }
        double best = 0.0;
        for (double v : dp) {
            if (std::isfinite(v) && v > best) {
                best = v;
            }
        }
        return best;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<uint64_t, double>& a, const std::pair<uint64_t, double>& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return __builtin_popcountll(a.first) > __builtin_popcountll(b.first);
              });
    uint64_t used = 0ULL;
    double total = 0.0;
    for (const auto& cand : candidates) {
        if ((cand.first & used) != 0ULL) {
            continue;
        }
        used |= cand.first;
        total += cand.second;
    }
    return total;
}

static EvalWithOpt EvaluateF_Original_WithOpt(
    IloEnv env,
    int zMask,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<int, char>& cacheMaskOpt,
    bool useCache
);

static double EvaluateF_Original_LP(
    IloEnv env,
    int zMask,
    std::unordered_map<int, double>& cacheMask,
    bool useCache
);

static int ApplySubmodularFixingForSupersets(
    const std::unordered_map<int, double>& cacheMask,
    const std::unordered_map<int, char>& cacheMaskOpt,
    int s1_mask,
    int s0_mask,
    int& addMask
) {
    addMask = 0;
    int freeMask = ((N_pn >= 31) ? -1 : ((1 << N_pn) - 1)) & ~(s0_mask | s1_mask);
    if (freeMask == 0) {
        return 0;
    }
    struct SubsetVal {
        int mask;
        double val;
    };
    std::vector<SubsetVal> subsets;
    subsets.reserve(cacheMaskOpt.size());
    for (const auto& kv : cacheMaskOpt) {
        if (kv.second == 0) continue;
        int T = kv.first;
        if ((T & ~s1_mask) != 0) continue; // only T' ⊆ s1_mask
        auto itT = cacheMask.find(T);
        if (itT == cacheMask.end()) continue;
        subsets.push_back({T, itT->second});
    }
    if (subsets.empty()) {
        return 0;
    }

    int remaining = freeMask & ~addMask;
    while (remaining) {
        int j = __builtin_ctz(static_cast<unsigned int>(remaining));
        int bit = (1 << j);
        remaining &= ~bit;
        bool fixJ = false;
        for (const auto& entry : subsets) {
            int T = entry.mask;
            double fT = entry.val;
            int sup = T | bit;
            auto itSup = cacheMask.find(sup);
            if (itSup == cacheMask.end()) {
                continue;
            }
            auto itOptSup = cacheMaskOpt.find(sup);
            if (itOptSup == cacheMaskOpt.end() || itOptSup->second == 0) {
                continue;
            }
            double fSup = itSup->second;
            if (!std::isfinite(fT) || !std::isfinite(fSup)) {
                continue;
            }
            if (fT + 1e-9 < fSup) {
                continue;
            }
            if ((fT - fSup) <= alpha[j] + 1e-9) {
                fixJ = true;
                break;
            }
        }
        if (fixJ) {
            addMask |= bit;
        }
    }
    return __builtin_popcount(static_cast<unsigned int>(addMask));
}

static int ChooseBranchVar_Option1(
    int s0_eff,
    int s1_eff,
    int sMax,
    double fS,
    const std::unordered_map<int, double>& cacheMask
) {
    const int allMask = (N_pn >= 31) ? -1 : ((1 << N_pn) - 1);
    int freeMask = allMask & ~(s0_eff | s1_eff);
    if (freeMask == 0) return -1;

    std::vector<int> cand;
    std::vector<double> dplus;
    std::vector<double> dminus;
    for (int j = 0; j < N_pn; ++j) {
        if (((freeMask >> j) & 1) == 0) continue;
        cand.push_back(j);
        dplus.push_back(-alpha[j]);
        double dm = 0.0;
        int sMinus = sMax & ~(1 << j);
        auto it = cacheMask.find(sMinus);
        if (it != cacheMask.end() && std::isfinite(fS) && std::isfinite(it->second)) {
            dm = fS - it->second;
        }
        dminus.push_back(dm);
    }
    if (cand.empty()) return -1;

    double minPlus = *std::min_element(dplus.begin(), dplus.end());
    double maxPlus = *std::max_element(dplus.begin(), dplus.end());
    double minMinus = *std::min_element(dminus.begin(), dminus.end());
    double maxMinus = *std::max_element(dminus.begin(), dminus.end());
    double denomPlus = maxPlus - minPlus;
    double denomMinus = maxMinus - minMinus;

    double bestPsi = -1e100;
    int bestJ = -1;
    for (size_t i = 0; i < cand.size(); ++i) {
        double zplus = (denomPlus > 1e-12) ? ((dplus[i] - minPlus) / denomPlus) : 0.0;
        double zminus = (denomMinus > 1e-12) ? ((dminus[i] - minMinus) / denomMinus) : 0.0;
        double psi = zplus + zminus;
        if (psi > bestPsi + 1e-12) {
            bestPsi = psi;
            bestJ = cand[i];
        }
    }
    return bestJ;
}

static int ChooseBranchVar_Option1_RL(
    int s0_eff,
    int s1_eff,
    int sMax,
    double fS,
    const std::unordered_map<int, RLObjective>& cacheMask
) {
    const int allMask = (N_pn >= 31) ? -1 : ((1 << N_pn) - 1);
    int freeMask = allMask & ~(s0_eff | s1_eff);
    if (freeMask == 0) return -1;

    std::vector<int> cand;
    std::vector<double> dplus;
    std::vector<double> dminus;
    for (int j = 0; j < N_pn; ++j) {
        if (((freeMask >> j) & 1) == 0) continue;
        cand.push_back(j);
        dplus.push_back(-alpha[j]);
        double dm = 0.0;
        int sMinus = sMax & ~(1 << j);
        auto it = cacheMask.find(sMinus);
        if (it != cacheMask.end() && std::isfinite(fS) && std::isfinite(it->second.ub)) {
            dm = fS - it->second.ub;
        }
        dminus.push_back(dm);
    }
    if (cand.empty()) return -1;

    double minPlus = *std::min_element(dplus.begin(), dplus.end());
    double maxPlus = *std::max_element(dplus.begin(), dplus.end());
    double minMinus = *std::min_element(dminus.begin(), dminus.end());
    double maxMinus = *std::max_element(dminus.begin(), dminus.end());
    double denomPlus = maxPlus - minPlus;
    double denomMinus = maxMinus - minMinus;

    double bestPsi = -1e100;
    int bestJ = -1;
    for (size_t i = 0; i < cand.size(); ++i) {
        double zplus = (denomPlus > 1e-12) ? ((dplus[i] - minPlus) / denomPlus) : 0.0;
        double zminus = (denomMinus > 1e-12) ? ((dminus[i] - minMinus) / denomMinus) : 0.0;
        double psi = zplus + zminus;
        if (psi > bestPsi + 1e-12) {
            bestPsi = psi;
            bestJ = cand[i];
        }
    }
    return bestJ;
}

class TauItemValidIneqCallback : public IloCplex::Callback::Function {
private:
    IloNumVarArray Y;
    IloNumVarArray2 I;
    IloNumVarArray2 L;
    std::vector<int> pbList;
    std::vector<std::vector<double>> wVal;
    int item;
    int cutCountPath;
    int cutCountTree;
public:
    TauItemValidIneqCallback(IloNumVarArray _Y, IloNumVarArray2 _I, IloNumVarArray2 _L,
                             const std::vector<int>& _pbList, int _item, int zMask)
        : Y(_Y), I(_I), L(_L), pbList(_pbList), item(_item),
          cutCountPath(0), cutCountTree(0) {
        const int pbCount = static_cast<int>(pbList.size());
        wVal.assign(pbCount, std::vector<double>(pbCount, 0.0));
        for (int a = 0; a < pbCount; ++a) {
            for (int b = 0; b < pbCount; ++b) {
                if (a == b) continue;
                double sum = 0.0;
                int pb1 = pbList[a];
                int pb2 = pbList[b];
                for (int j = 0; j < N_pn; ++j) {
                    if ((zMask >> j) & 1) {
                        sum += std::fabs(eta[j][pb1] - eta[j][pb2]);
                    }
                }
                wVal[a][b] = sum;
            }
        }
    }

    void invoke(const IloCplex::Callback::Context& context) override {
        if (!context.inRelaxation()) return;
        if (context.getIntInfo(IloCplex::Callback::Context::Info::NodeDepth) != 0) return;
        if (VI_pro1 == 0 && VI_pro2 == 0) return;

        const int pbCount = static_cast<int>(pbList.size());
        if (pbCount == 0) return;

        IloEnv env = context.getEnv();
        std::vector<double> YVals(N_tp + 1, 0.0);
        for (int t = 0; t <= N_tp; ++t) {
            YVals[t] = context.getRelaxationPoint(Y[t]);
        }

        std::vector<std::vector<std::vector<double>>> IVals(
            N_sc, std::vector<std::vector<double>>(pbCount, std::vector<double>(N_tp + 1, 0.0)));
        std::vector<std::vector<std::vector<double>>> LVals(
            N_sc, std::vector<std::vector<double>>(pbCount, std::vector<double>(N_tp + 1, 0.0)));
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                for (int t = 0; t <= N_tp; ++t) {
                    int idx = pbIdx * (N_tp + 1) + t;
                    IVals[sc][pbIdx][t] = context.getRelaxationPoint(I[sc][idx]);
                    LVals[sc][pbIdx][t] = context.getRelaxationPoint(L[sc][idx]);
                }
            }
        }

        if (VI_pro1 == 1) {
            for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                int pb = pbList[pbIdx];
                int k = 1;
                while (k < N_tp) {
                    int bestSc = -1;
                    double bestViol = 0.0;
                    for (int sc = 0; sc < N_sc; ++sc) {
                        double viol = IVals[sc][pbIdx][k];
                        double valSumY = 0.0;
                        for (int v = k + 1; v <= N_tp; ++v) {
                            valSumY += YVals[v];
                            if (valSumY >= 1.0) break;
                            double slack = d[v][sc][pb][item] * (1.0 - valSumY) - LVals[sc][pbIdx][v];
                            if (slack > 0.1) {
                                viol -= d[v][sc][pb][item] * (1.0 - valSumY);
                                viol += LVals[sc][pbIdx][v];
                            }
                        }
                        if (viol < bestViol) {
                            bestViol = viol;
                            bestSc = sc;
                        }
                    }
                    if (bestViol < -0.1 && bestSc >= 0) {
                        std::vector<int> listU;
                        int maxU = 0;
                        double valSumY = 0.0;
                        for (int v = k + 1; v <= N_tp; ++v) {
                            valSumY += YVals[v];
                            if (valSumY >= 1.0) break;
                            double slack = d[v][bestSc][pb][item] * (1.0 - valSumY) - LVals[bestSc][pbIdx][v];
                            if (slack > 0.1) {
                                listU.push_back(v);
                                maxU = v;
                            }
                        }
                        if (!listU.empty()) {
                            IloExpr cutExpr(env);
                            cutExpr = I[bestSc][pbIdx * (N_tp + 1) + k];
                            for (int v : listU) {
                                cutExpr += L[bestSc][pbIdx * (N_tp + 1) + v];
                                cutExpr += -d[v][bestSc][pb][item];
                            }
                            for (int l = k + 1; l <= maxU; ++l) {
                                double auxD = 0.0;
                                for (int v : listU) {
                                    if (v >= l) auxD += d[v][bestSc][pb][item];
                                }
                                cutExpr += std::min(static_cast<double>(Cap), auxD) * Y[l];
                            }
                            context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                            cutCountPath++;
                            VIPathCount++;
                            cutExpr.end();
                            k = maxU + 1;
                        } else {
                            k++;
                        }
                    } else {
                        k++;
                    }
                }
            }
        }

        if (VI_pro2 == 1 && pbCount > 1) {
            for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                int pb1 = pbList[pbIdx];
                for (int pb2Idx = 0; pb2Idx < pbCount; ++pb2Idx) {
                    if (pb2Idx == pbIdx) continue;
                    for (int k = 1; k < N_tp; ++k) {
                        std::vector<std::vector<int>> U_sc(N_sc);
                        std::vector<double> D_sc(N_sc, 0.0);
                        std::vector<int> t_sc(N_sc, k);
                        std::vector<std::vector<double>> Dsc_v(N_sc, std::vector<double>(N_tp + 1, 0.0));
                        std::vector<char> inUnion(N_tp + 1, 0);
                        std::vector<int> unionU;

                        for (int sc = 0; sc < N_sc; ++sc) {
                            int maxU = 0;
                            double valSumY = 0.0;
                            for (int v = k + 1; v <= N_tp; ++v) {
                                valSumY += YVals[v];
                                if (valSumY >= 1.0) break;
                                double slack = d[v][sc][pb1][item] * (1.0 - valSumY) - LVals[sc][pbIdx][v];
                                if (slack > 0.1) {
                                    U_sc[sc].push_back(v);
                                    maxU = v;
                                }
                            }
                            if (maxU <= k) continue;
                            t_sc[sc] = maxU;
                            double sumD = 0.0;
                            for (int v : U_sc[sc]) {
                                sumD += d[v][sc][pb1][item];
                                if (!inUnion[v]) {
                                    inUnion[v] = 1;
                                    unionU.push_back(v);
                                }
                            }
                            D_sc[sc] = sumD;
                            double suffix = 0.0;
                            for (int v = maxU; v >= k + 1; --v) {
                                suffix += d[v][sc][pb1][item];
                                Dsc_v[sc][v] = suffix;
                            }
                        }

                        if (unionU.empty()) continue;
                        std::sort(unionU.begin(), unionU.end());

                        std::vector<int> order(N_sc);
                        for (int sc = 0; sc < N_sc; ++sc) order[sc] = sc;
                        std::sort(order.begin(), order.end(), [&](int a, int b) {
                            if (D_sc[a] == D_sc[b]) return a < b;
                            return D_sc[a] < D_sc[b];
                        });
                        int scMax = order.back();
                        double D_max = D_sc[scMax];
                        int t_max = t_sc[scMax];
                        if (D_max <= 0.0 || t_max <= k) continue;

                        std::vector<double> phi(N_tp + 1, 0.0);
                        for (int v = k + 1; v <= t_max; ++v) {
                            int l = -1;
                            for (int idx = N_sc - 1; idx >= 0; --idx) {
                                int sc = order[idx];
                                if (t_sc[sc] >= v) { l = idx; break; }
                            }
                            if (l < 0) continue;
                            double D_lv = Dsc_v[order[l]][v];
                            double second = D_max - D_lv - 1.0;
                            double best = std::numeric_limits<double>::infinity();
                            for (int idx = 0; idx < N_sc; ++idx) {
                                int sc = order[idx];
                                if (t_sc[sc] < v) continue;
                                double cand = std::max(Dsc_v[sc][v], second);
                                if (cand < best) best = cand;
                            }
                            if (std::isfinite(best) && best > 0.0) phi[v] = best;
                        }

                        int bestSc = -1;
                        double bestViol = 0.0;
                        for (int sc = 0; sc < N_sc; ++sc) {
                            double lhs = IVals[sc][pbIdx][k];
                            for (int v : unionU) lhs += LVals[sc][pbIdx][v];
                            for (int v = k + 1; v <= t_max; ++v) {
                                if (phi[v] > 0.0) lhs += phi[v] * YVals[v];
                            }
                            lhs += D_max * wVal[pbIdx][pb2Idx];
                            double viol = lhs - D_max;
                            if (viol < bestViol) {
                                bestViol = viol;
                                bestSc = sc;
                            }
                        }
                        if (bestViol < -0.1 && bestSc >= 0) {
                            IloExpr cutExpr(env);
                            cutExpr = I[bestSc][pbIdx * (N_tp + 1) + k];
                            for (int v : unionU) {
                                cutExpr += L[bestSc][pbIdx * (N_tp + 1) + v];
                            }
                            for (int v = k + 1; v <= t_max; ++v) {
                                if (phi[v] > 0.0) cutExpr += phi[v] * Y[v];
                            }
                            cutExpr += D_max * wVal[pbIdx][pb2Idx];
                            cutExpr += -D_max;
                            context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                            cutCountTree++;
                            VITreeCount++;
                            cutExpr.end();
                        }
                    }
                }
            }
        }
    }
};

class TauJointValidIneqCallback : public IloCplex::Callback::Function {
private:
    IloNumVarArray2 Y;
    IloNumVarArray3 I;
    IloNumVarArray3 L;
    std::vector<int> pbList;
    std::vector<std::vector<double>> wVal;
public:
    TauJointValidIneqCallback(IloNumVarArray2 _Y, IloNumVarArray3 _I, IloNumVarArray3 _L,
                              const std::vector<int>& _pbList, int zMask)
        : Y(_Y), I(_I), L(_L), pbList(_pbList) {
        const int pbCount = static_cast<int>(pbList.size());
        wVal.assign(pbCount, std::vector<double>(pbCount, 0.0));
        for (int a = 0; a < pbCount; ++a) {
            for (int b = 0; b < pbCount; ++b) {
                if (a == b) continue;
                double sum = 0.0;
                int pb1 = pbList[a];
                int pb2 = pbList[b];
                for (int j = 0; j < N_pn; ++j) {
                    if ((zMask >> j) & 1) {
                        sum += std::fabs(eta[j][pb1] - eta[j][pb2]);
                    }
                }
                wVal[a][b] = sum;
            }
        }
    }

    void invoke(const IloCplex::Callback::Context& context) override {
        if (!context.inRelaxation()) return;
        if (context.getIntInfo(IloCplex::Callback::Context::Info::NodeDepth) != 0) return;
        if (VI_pro1 == 0 && VI_pro2 == 0) return;

        const int pbCount = static_cast<int>(pbList.size());
        if (pbCount == 0) return;

        IloEnv env = context.getEnv();

        std::vector<std::vector<double>> YVals(N_it, std::vector<double>(N_tp + 1, 0.0));
        for (int it = 0; it < N_it; ++it) {
            for (int t = 0; t <= N_tp; ++t) {
                YVals[it][t] = context.getRelaxationPoint(Y[it][t]);
            }
        }

        std::vector<std::vector<std::vector<std::vector<double>>>> IVals(
            N_it, std::vector<std::vector<std::vector<double>>>(N_sc,
            std::vector<std::vector<double>>(pbCount, std::vector<double>(N_tp + 1, 0.0))));
        std::vector<std::vector<std::vector<std::vector<double>>>> LVals(
            N_it, std::vector<std::vector<std::vector<double>>>(N_sc,
            std::vector<std::vector<double>>(pbCount, std::vector<double>(N_tp + 1, 0.0))));
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                    for (int t = 0; t <= N_tp; ++t) {
                        int idx = pbIdx * (N_tp + 1) + t;
                        IVals[it][sc][pbIdx][t] = context.getRelaxationPoint(I[it][sc][idx]);
                        LVals[it][sc][pbIdx][t] = context.getRelaxationPoint(L[it][sc][idx]);
                    }
                }
            }
        }

        for (int it = 0; it < N_it; ++it) {
            if (VI_pro1 == 1) {
                for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                    int pb = pbList[pbIdx];
                    int k = 1;
                    while (k < N_tp) {
                        int bestSc = -1;
                        double bestViol = 0.0;
                        for (int sc = 0; sc < N_sc; ++sc) {
                            double viol = IVals[it][sc][pbIdx][k];
                            double valSumY = 0.0;
                            for (int v = k + 1; v <= N_tp; ++v) {
                                valSumY += YVals[it][v];
                                if (valSumY >= 1.0) break;
                                double slack = d[v][sc][pb][it] * (1.0 - valSumY) - LVals[it][sc][pbIdx][v];
                                if (slack > 0.1) {
                                    viol -= d[v][sc][pb][it] * (1.0 - valSumY);
                                    viol += LVals[it][sc][pbIdx][v];
                                }
                            }
                            if (viol < bestViol) {
                                bestViol = viol;
                                bestSc = sc;
                            }
                        }
                        if (bestViol < -0.1 && bestSc >= 0) {
                            std::vector<int> listU;
                            int maxU = 0;
                            double valSumY = 0.0;
                            for (int v = k + 1; v <= N_tp; ++v) {
                                valSumY += YVals[it][v];
                                if (valSumY >= 1.0) break;
                                double slack = d[v][bestSc][pb][it] * (1.0 - valSumY) - LVals[it][bestSc][pbIdx][v];
                                if (slack > 0.1) {
                                    listU.push_back(v);
                                    maxU = v;
                                }
                            }
                            if (!listU.empty()) {
                                IloExpr cutExpr(env);
                                cutExpr = I[it][bestSc][pbIdx * (N_tp + 1) + k];
                                for (int v : listU) {
                                    cutExpr += L[it][bestSc][pbIdx * (N_tp + 1) + v];
                                    cutExpr += -d[v][bestSc][pb][it];
                                }
                                for (int l = k + 1; l <= maxU; ++l) {
                                    double auxD = 0.0;
                                    for (int v : listU) {
                                        if (v >= l) auxD += d[v][bestSc][pb][it];
                                    }
                                    cutExpr += std::min(static_cast<double>(Cap), auxD) * Y[it][l];
                                }
                                context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                                VIPathCount++;
                                cutExpr.end();
                                k = maxU + 1;
                            } else {
                                k++;
                            }
                        } else {
                            k++;
                        }
                    }
                }
            }

            if (VI_pro2 == 1 && pbCount > 1) {
                for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
                    int pb1 = pbList[pbIdx];
                    for (int pb2Idx = 0; pb2Idx < pbCount; ++pb2Idx) {
                        if (pb2Idx == pbIdx) continue;
                        for (int k = 1; k < N_tp; ++k) {
                            std::vector<std::vector<int>> U_sc(N_sc);
                            std::vector<double> D_sc(N_sc, 0.0);
                            std::vector<int> t_sc(N_sc, k);
                            std::vector<std::vector<double>> Dsc_v(N_sc, std::vector<double>(N_tp + 1, 0.0));
                            std::vector<char> inUnion(N_tp + 1, 0);
                            std::vector<int> unionU;

                            for (int sc = 0; sc < N_sc; ++sc) {
                                int maxU = 0;
                                double valSumY = 0.0;
                                for (int v = k + 1; v <= N_tp; ++v) {
                                    valSumY += YVals[it][v];
                                    if (valSumY >= 1.0) break;
                                    double slack = d[v][sc][pb1][it] * (1.0 - valSumY) - LVals[it][sc][pbIdx][v];
                                    if (slack > 0.1) {
                                        U_sc[sc].push_back(v);
                                        maxU = v;
                                    }
                                }
                                if (maxU <= k) continue;
                                t_sc[sc] = maxU;
                                double sumD = 0.0;
                                for (int v : U_sc[sc]) {
                                    sumD += d[v][sc][pb1][it];
                                    if (!inUnion[v]) {
                                        inUnion[v] = 1;
                                        unionU.push_back(v);
                                    }
                                }
                                D_sc[sc] = sumD;
                                double suffix = 0.0;
                                for (int v = maxU; v >= k + 1; --v) {
                                    suffix += d[v][sc][pb1][it];
                                    Dsc_v[sc][v] = suffix;
                                }
                            }

                            if (unionU.empty()) continue;
                            std::sort(unionU.begin(), unionU.end());

                            std::vector<int> order(N_sc);
                            for (int sc = 0; sc < N_sc; ++sc) order[sc] = sc;
                            std::sort(order.begin(), order.end(), [&](int a, int b) {
                                if (D_sc[a] == D_sc[b]) return a < b;
                                return D_sc[a] < D_sc[b];
                            });
                            int scMax = order.back();
                            double D_max = D_sc[scMax];
                            int t_max = t_sc[scMax];
                            if (D_max <= 0.0 || t_max <= k) continue;

                            std::vector<double> phi(N_tp + 1, 0.0);
                            for (int v = k + 1; v <= t_max; ++v) {
                                int l = -1;
                                for (int idx = N_sc - 1; idx >= 0; --idx) {
                                    int sc = order[idx];
                                    if (t_sc[sc] >= v) { l = idx; break; }
                                }
                                if (l < 0) continue;
                                double D_lv = Dsc_v[order[l]][v];
                                double second = D_max - D_lv - 1.0;
                                double best = std::numeric_limits<double>::infinity();
                                for (int idx = 0; idx < N_sc; ++idx) {
                                    int sc = order[idx];
                                    if (t_sc[sc] < v) continue;
                                    double cand = std::max(Dsc_v[sc][v], second);
                                    if (cand < best) best = cand;
                                }
                                if (std::isfinite(best) && best > 0.0) phi[v] = best;
                            }

                            int bestSc = -1;
                            double bestViol = 0.0;
                            for (int sc = 0; sc < N_sc; ++sc) {
                                double lhs = IVals[it][sc][pbIdx][k];
                                for (int v : unionU) lhs += LVals[it][sc][pbIdx][v];
                                for (int v = k + 1; v <= t_max; ++v) {
                                    if (phi[v] > 0.0) lhs += phi[v] * YVals[it][v];
                                }
                                lhs += D_max * wVal[pbIdx][pb2Idx];
                                double viol = lhs - D_max;
                                if (viol < bestViol) {
                                    bestViol = viol;
                                    bestSc = sc;
                                }
                            }
                            if (bestViol < -0.1 && bestSc >= 0) {
                                IloExpr cutExpr(env);
                                cutExpr = I[it][bestSc][pbIdx * (N_tp + 1) + k];
                                for (int v : unionU) {
                                    cutExpr += L[it][bestSc][pbIdx * (N_tp + 1) + v];
                                }
                                for (int v = k + 1; v <= t_max; ++v) {
                                    if (phi[v] > 0.0) cutExpr += phi[v] * Y[it][v];
                                }
                                cutExpr += D_max * wVal[pbIdx][pb2Idx];
                                cutExpr += -D_max;
                            context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                            VITreeCount++;
                            cutExpr.end();
                            }
                        }
                    }
                }
            }
        }
    }
};

class OriginalValidIneqCallback : public IloCplex::Callback::Function {
private:
    IloNumVarArray3 Y;
    IloNumVarArray4 I;
    IloNumVarArray4 L;
    std::vector<int> pbToTau;
    std::vector<std::vector<double>> wVal;
public:
    OriginalValidIneqCallback(IloNumVarArray3 _Y, IloNumVarArray4 _I, IloNumVarArray4 _L,
                              const std::vector<int>& _pbToTau, int zMask)
        : Y(_Y), I(_I), L(_L), pbToTau(_pbToTau) {
        wVal.assign(N_pb, std::vector<double>(N_pb, 0.0));
        for (int a = 0; a < N_pb; ++a) {
            for (int b = 0; b < N_pb; ++b) {
                if (a == b) continue;
                double sum = 0.0;
                for (int j = 0; j < N_pn; ++j) {
                    if ((zMask >> j) & 1) {
                        sum += std::fabs(eta[j][a] - eta[j][b]);
                    }
                }
                wVal[a][b] = sum;
            }
        }
    }

    void invoke(const IloCplex::Callback::Context& context) override {
        if (!context.inRelaxation()) return;
        if (context.getIntInfo(IloCplex::Callback::Context::Info::NodeDepth) != 0) return;
        if (VI_pro1 == 0 && VI_pro2 == 0) return;

        IloEnv env = context.getEnv();

        std::vector<std::vector<std::vector<double>>> YVals(
            N_it, std::vector<std::vector<double>>(N_pb, std::vector<double>(N_tp + 1, 0.0)));
        for (int it = 0; it < N_it; ++it) {
            for (int pb = 0; pb < N_pb; ++pb) {
                int tauIdx = pbToTau[pb];
                for (int t = 0; t <= N_tp; ++t) {
                    YVals[it][pb][t] = context.getRelaxationPoint(Y[it][tauIdx][t]);
                }
            }
        }

        std::vector<std::vector<std::vector<std::vector<double>>>> IVals(
            N_it, std::vector<std::vector<std::vector<double>>>(N_sc,
            std::vector<std::vector<double>>(N_pb, std::vector<double>(N_tp + 1, 0.0))));
        std::vector<std::vector<std::vector<std::vector<double>>>> LVals(
            N_it, std::vector<std::vector<std::vector<double>>>(N_sc,
            std::vector<std::vector<double>>(N_pb, std::vector<double>(N_tp + 1, 0.0))));
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int pb = 0; pb < N_pb; ++pb) {
                    for (int t = 0; t <= N_tp; ++t) {
                        IVals[it][sc][pb][t] = context.getRelaxationPoint(I[it][sc][pb][t]);
                        LVals[it][sc][pb][t] = context.getRelaxationPoint(L[it][sc][pb][t]);
                    }
                }
            }
        }

        for (int it = 0; it < N_it; ++it) {
            if (VI_pro1 == 1) {
                for (int pb = 0; pb < N_pb; ++pb) {
                    int k = 1;
                    while (k < N_tp) {
                        int bestSc = -1;
                        double bestViol = 0.0;
                        for (int sc = 0; sc < N_sc; ++sc) {
                            double viol = IVals[it][sc][pb][k];
                            double valSumY = 0.0;
                            for (int v = k + 1; v <= N_tp; ++v) {
                                valSumY += YVals[it][pb][v];
                                if (valSumY >= 1.0) break;
                                double slack = d[v][sc][pb][it] * (1.0 - valSumY) - LVals[it][sc][pb][v];
                                if (slack > 0.1) {
                                    viol -= d[v][sc][pb][it] * (1.0 - valSumY);
                                    viol += LVals[it][sc][pb][v];
                                }
                            }
                            if (viol < bestViol) {
                                bestViol = viol;
                                bestSc = sc;
                            }
                        }
                        if (bestViol < -0.1 && bestSc >= 0) {
                            std::vector<int> listU;
                            int maxU = 0;
                            double valSumY = 0.0;
                            for (int v = k + 1; v <= N_tp; ++v) {
                                valSumY += YVals[it][pb][v];
                                if (valSumY >= 1.0) break;
                                double slack = d[v][bestSc][pb][it] * (1.0 - valSumY) - LVals[it][bestSc][pb][v];
                                if (slack > 0.1) {
                                    listU.push_back(v);
                                    maxU = v;
                                }
                            }
                            if (!listU.empty()) {
                                IloExpr cutExpr(env);
                                cutExpr = I[it][bestSc][pb][k];
                                for (int v : listU) {
                                    cutExpr += L[it][bestSc][pb][v];
                                    cutExpr += -d[v][bestSc][pb][it];
                                }
                                for (int l = k + 1; l <= maxU; ++l) {
                                    double auxD = 0.0;
                                    for (int v : listU) {
                                        if (v >= l) auxD += d[v][bestSc][pb][it];
                                    }
                                    cutExpr += std::min(static_cast<double>(Cap), auxD) * Y[it][pbToTau[pb]][l];
                                }
                                context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                                VIPathCount++;
                                cutExpr.end();
                                k = maxU + 1;
                            } else {
                                k++;
                            }
                        } else {
                            k++;
                        }
                    }
                }
            }

            if (VI_pro2 == 1 && N_pb > 1) {
                for (int pb1 = 0; pb1 < N_pb; ++pb1) {
                    for (int pb2 = 0; pb2 < N_pb; ++pb2) {
                        if (pb2 == pb1) continue;
                        for (int k = 1; k < N_tp; ++k) {
                            std::vector<std::vector<int>> U_sc(N_sc);
                            std::vector<double> D_sc(N_sc, 0.0);
                            std::vector<int> t_sc(N_sc, k);
                            std::vector<std::vector<double>> Dsc_v(N_sc, std::vector<double>(N_tp + 1, 0.0));
                            std::vector<char> inUnion(N_tp + 1, 0);
                            std::vector<int> unionU;

                            for (int sc = 0; sc < N_sc; ++sc) {
                                int maxU = 0;
                                double valSumY = 0.0;
                                for (int v = k + 1; v <= N_tp; ++v) {
                                    valSumY += YVals[it][pb2][v];
                                    if (valSumY >= 1.0) break;
                                    double slack = d[v][sc][pb1][it] * (1.0 - valSumY) - LVals[it][sc][pb1][v];
                                    if (slack > 0.1) {
                                        U_sc[sc].push_back(v);
                                        maxU = v;
                                    }
                                }
                                if (maxU <= k) continue;
                                t_sc[sc] = maxU;
                                double sumD = 0.0;
                                for (int v : U_sc[sc]) {
                                    sumD += d[v][sc][pb1][it];
                                    if (!inUnion[v]) {
                                        inUnion[v] = 1;
                                        unionU.push_back(v);
                                    }
                                }
                                D_sc[sc] = sumD;
                                double suffix = 0.0;
                                for (int v = maxU; v >= k + 1; --v) {
                                    suffix += d[v][sc][pb1][it];
                                    Dsc_v[sc][v] = suffix;
                                }
                            }

                            if (unionU.empty()) continue;
                            std::sort(unionU.begin(), unionU.end());

                            std::vector<int> order(N_sc);
                            for (int sc = 0; sc < N_sc; ++sc) order[sc] = sc;
                            std::sort(order.begin(), order.end(), [&](int a, int b) {
                                if (D_sc[a] == D_sc[b]) return a < b;
                                return D_sc[a] < D_sc[b];
                            });
                            int scMax = order.back();
                            double D_max = D_sc[scMax];
                            int t_max = t_sc[scMax];
                            if (D_max <= 0.0 || t_max <= k) continue;

                            std::vector<double> phi(N_tp + 1, 0.0);
                            for (int v = k + 1; v <= t_max; ++v) {
                                int l = -1;
                                for (int idx = N_sc - 1; idx >= 0; --idx) {
                                    int sc = order[idx];
                                    if (t_sc[sc] >= v) { l = idx; break; }
                                }
                                if (l < 0) continue;
                                double D_lv = Dsc_v[order[l]][v];
                                double second = D_max - D_lv - 1.0;
                                double best = std::numeric_limits<double>::infinity();
                                for (int idx = 0; idx < N_sc; ++idx) {
                                    int sc = order[idx];
                                    if (t_sc[sc] < v) continue;
                                    double cand = std::max(Dsc_v[sc][v], second);
                                    if (cand < best) best = cand;
                                }
                                if (std::isfinite(best) && best > 0.0) phi[v] = best;
                            }

                            int bestSc = -1;
                            double bestViol = 0.0;
                            for (int sc = 0; sc < N_sc; ++sc) {
                                double lhs = IVals[it][sc][pb1][k];
                                for (int v : unionU) lhs += LVals[it][sc][pb1][v];
                                for (int v = k + 1; v <= t_max; ++v) {
                                    if (phi[v] > 0.0) lhs += phi[v] * YVals[it][pb2][v];
                                }
                                lhs += D_max * wVal[pb1][pb2];
                                double viol = lhs - D_max;
                                if (viol < bestViol) {
                                    bestViol = viol;
                                    bestSc = sc;
                                }
                            }
                            if (bestViol < -0.1 && bestSc >= 0) {
                                IloExpr cutExpr(env);
                                cutExpr = I[it][bestSc][pb1][k];
                                for (int v : unionU) {
                                    cutExpr += L[it][bestSc][pb1][v];
                                }
                                for (int v = k + 1; v <= t_max; ++v) {
                                    if (phi[v] > 0.0) cutExpr += phi[v] * Y[it][pbToTau[pb2]][v];
                                }
                                cutExpr += D_max * wVal[pb1][pb2];
                                cutExpr += -D_max;
                                context.addUserCut(cutExpr >= 0, IloCplex::UseCutForce, IloTrue);
                                VITreeCount++;
                                cutExpr.end();
                            }
                        }
                    }
                }
            }
        }
    }
};

static double SolveTauItemSubproblem(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    int zMask
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= min(Cap, minBigM) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                int idxPrev = k * (N_tp+1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp+1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauItemValidIneqCallback(Y, I, L, pbList, item, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauItemSubproblem_WithSol(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    int zMask,
    std::vector<int>& ySol,
    std::vector<double>& xSol
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= min(Cap, minBigM) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                int idxPrev = k * (N_tp+1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp+1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauItemValidIneqCallback(Y, I, L, pbList, item, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return IloInfinity;
    }

    ySol.assign(N_tp + 1, 0);
    xSol.assign(N_tp + 1, 0.0);
    for (int t = 0; t <= N_tp; ++t) {
        xSol[t] = cplex.getValue(X[t]);
        double yv = cplex.getValue(Y[t]);
        ySol[t] = (yv >= 0.5) ? 1 : 0;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauItemSubproblem_Cap(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<double>& capIt
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= min(Cap, minBigM) * Y[t]);
        model.add(X[t] <= capIt[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                int idxPrev = k * (N_tp+1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp+1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauItemSubproblem_Cap_WithSol(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<double>& capIt,
    int zMask,
    std::vector<int>& ySol,
    std::vector<double>& xSol,
    int cplexThreads
){
    const int pbCount = static_cast<int>(pbList.size());
    ySol.assign(N_tp + 1, 0);
    xSol.assign(N_tp + 1, 0.0);
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= min(Cap, minBigM) * Y[t]);
        double capBound = (t < static_cast<int>(capIt.size())) ? capIt[t] : static_cast<double>(Cap);
        if (capBound < 0.0) capBound = 0.0;
        model.add(X[t] <= capBound);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                int idxPrev = k * (N_tp+1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp+1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, std::max(1, cplexThreads));
    cplex.setParam(IloCplex::TiLim, 120);
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauItemValidIneqCallback(Y, I, L, pbList, item, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    for (int t = 0; t <= N_tp; ++t) {
        xSol[t] = cplex.getValue(X[t]);
        double yv = cplex.getValue(Y[t]);
        ySol[t] = (yv >= 0.5) ? 1 : 0;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double PhiItemCapCached(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<int>& kVec,
    std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>& cache
){
    auto& map = cache[item];
    auto it = map.find(kVec);
    if (it != map.end()) {
        return it->second;
    }
    std::vector<double> capIt(N_tp + 1, 0.0);
    for (int t = 1; t <= N_tp; ++t) {
        capIt[t] = static_cast<double>(kVec[t]);
    }
    double val = SolveTauItemSubproblem_Cap(env, item, pbList, capIt);
    map.emplace(kVec, val);
    return val;
}

static bool CapBoundsFeasibleEq(const std::vector<int>& L, const std::vector<int>& U) {
    for (int t = 1; t <= N_tp; ++t) {
        int sumL = 0;
        int sumU = 0;
        for (int it = 0; it < N_it; ++it) {
            sumL += L[CapIdx(it, t)];
            sumU += U[CapIdx(it, t)];
        }
        if (sumL > Cap || sumU < Cap) {
            return false;
        }
    }
    return true;
}

static double ComputeUB_DiscreteCap(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<int>& L,
    const std::vector<int>& U,
    std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>& phiCache
){
    std::vector<std::vector<int>> k(N_it, std::vector<int>(N_tp + 1, 0));
    std::vector<int> sumK(N_tp + 1, 0);
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int val = L[CapIdx(it, t)];
            k[it][t] = val;
            sumK[t] += val;
        }
    }

    std::vector<double> phiCurrent(N_it, 0.0);
    for (int it = 0; it < N_it; ++it) {
        phiCurrent[it] = PhiItemCapCached(env, it, pbList, k[it], phiCache);
    }

    for (int t = 1; t <= N_tp; ++t) {
        int residual = Cap - sumK[t];
        if (residual <= 0) {
            continue;
        }
        while (residual > 0) {
            int bestItem = -1;
            double bestGain = -IloInfinity;
            double bestPhiPlus = 0.0;
            for (int it = 0; it < N_it; ++it) {
                int idx = CapIdx(it, t);
                if (k[it][t] >= U[idx]) {
                    continue;
                }
                std::vector<int> kPlus = k[it];
                kPlus[t] += 1;
                double phiPlus = PhiItemCapCached(env, it, pbList, kPlus, phiCache);
                double gain = phiCurrent[it] - phiPlus;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestItem = it;
                    bestPhiPlus = phiPlus;
                }
            }
            if (bestItem < 0) {
                return IloInfinity;
            }
            k[bestItem][t] += 1;
            sumK[t] += 1;
            residual -= 1;
            phiCurrent[bestItem] = bestPhiPlus;
        }
    }

    double ub = 0.0;
    for (int it = 0; it < N_it; ++it) {
        ub += phiCurrent[it];
    }
    return ub;
}

static double ComputeLB_DiscreteCap(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<int>& L,
    const std::vector<int>& U,
    std::vector<std::unordered_map<std::vector<int>, double, IntVecHash>>& phiCache
){
    std::vector<std::vector<int>> u(N_it, std::vector<int>(N_tp + 1, 0));
    std::vector<int> sumU(N_tp + 1, 0);
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int val = U[CapIdx(it, t)];
            u[it][t] = val;
            sumU[t] += val;
        }
    }

    std::vector<double> phiU(N_it, 0.0);
    for (int it = 0; it < N_it; ++it) {
        phiU[it] = PhiItemCapCached(env, it, pbList, u[it], phiCache);
    }

    double lb = 0.0;
    for (int it = 0; it < N_it; ++it) {
        lb += phiU[it];
    }

    for (int t = 1; t <= N_tp; ++t) {
        if (sumU[t] < Cap) {
            return IloInfinity;
        }
        int excess = sumU[t] - Cap;
        if (excess <= 0) {
            continue;
        }
        std::vector<double> marginals;
        for (int it = 0; it < N_it; ++it) {
            int u_it = U[CapIdx(it, t)];
            int l_it = L[CapIdx(it, t)];
            int removable = u_it - l_it;
            if (removable <= 0) {
                continue;
            }
            std::vector<int> k = u[it];
            double phiCur = phiU[it];
            for (int q = 0; q < removable; ++q) {
                k[t] -= 1;
                double phiNew = PhiItemCapCached(env, it, pbList, k, phiCache);
                marginals.push_back(phiNew - phiCur);
                phiCur = phiNew;
            }
        }
        if (excess > static_cast<int>(marginals.size())) {
            return IloInfinity;
        }
        std::sort(marginals.begin(), marginals.end());
        for (int k = 0; k < excess; ++k) {
            lb += marginals[k];
        }
    }
    return lb;
}

static double SolveTauJointSubproblem_BBCap(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    int tauRep,
    const std::string& tauKey,
    bool* optimalOut = nullptr
){
    if (pbList.empty()) {
        if (optimalOut) {
            *optimalOut = true;
        }
        return 0.0;
    }

    cout << "Capacity subproblem start tau=" << tauRep
         << " bestLB=" << IloInfinity
         << " bestUB=" << IloInfinity
         << " pbCount=" << pbList.size() << endl;

    std::vector<int> rootL(N_it * (N_tp + 1), 0);
    std::vector<int> rootU(N_it * (N_tp + 1), 0);
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            rootU[CapIdx(it, t)] = Cap;
        }
    }

    auto& phiCache = GlobalPhiCache[tauKey];
    if (phiCache.empty()) {
        phiCache.resize(N_it);
    }

    std::vector<CapBBNode> stack;
    CapBBNode root;
    root.L = rootL;
    root.U = rootU;
    root.lb = 0.0;
    root.ub = IloInfinity;
    stack.push_back(root);

    double bestUB = IloInfinity;
    int iter = 0;
    int pruned = 0;
    auto getGlobalLB = [&](double currentLb, bool includeCurrent) -> double {
        double minOpen = MinLBStack(stack);
        if (includeCurrent && std::isfinite(currentLb)) {
            if (currentLb < minOpen) {
                minOpen = currentLb;
            }
        }
        if (!std::isfinite(minOpen)) {
            return bestUB;
        }
        return minOpen;
    };

    while (!stack.empty()) {
        double minOpen = MinLBStack(stack);
        if (std::isfinite(bestUB) && std::isfinite(minOpen) && minOpen >= bestUB - 1e-9) {
            break;
        }
        CapBBNode node = std::move(stack.back());
        stack.pop_back();
        iter++;
        auto printNodePrefix = [&](double globalLB) {
            cout << "Capacity subproblem node " << iter
                 << " tau=" << tauRep
                 << " bestLB=" << globalLB
                 << " bestUB=" << bestUB;
        };

        if (!CapBoundsFeasibleEq(node.L, node.U)) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " infeasible sumL>Cap"
                 << " stack=" << stack.size() << endl;
            continue;
        }

        if (!PropagateU_ByResidual(node.L, node.U)) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " infeasible after propagate"
                 << " stack=" << stack.size() << endl;
            continue;
        }

        if (!PropagateL_BySumU(node.L, node.U)) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " infeasible after propagate"
                 << " stack=" << stack.size() << endl;
            continue;
        }

        if (!CapBoundsFeasibleEq(node.L, node.U)) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " infeasible sumU<Cap"
                 << " stack=" << stack.size() << endl;
            continue;
        }

        if (SumU_EqCapAll(node.U)) {
            double val = PhiSumAtU(env, pbList, node.U, phiCache);
            if (val < bestUB) {
                bestUB = val;
            }
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " closed sumU=Cap val=" << val
                 << " stack=" << stack.size() << endl;
            continue;
        }
        if (SumL_EqCapAll(node.L)) {
            double val = PhiSumAtU(env, pbList, node.L, phiCache);
            if (val < bestUB) {
                bestUB = val;
            }
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " closed sumL=Cap val=" << val
                 << " stack=" << stack.size() << endl;
            continue;
        }

        double lb = ComputeLB_DiscreteCap(env, pbList, node.L, node.U, phiCache);
        if (lb >= bestUB - 1e-9) {
            pruned++;
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " pruned lb=" << lb
                 << " stack=" << stack.size() << endl;
            continue;
        }

        double ub = ComputeUB_DiscreteCap(env, pbList, node.L, node.U, phiCache);
        if (ub < bestUB) {
            bestUB = ub;
            double globalLB = getGlobalLB(lb, true);
            printNodePrefix(globalLB);
            cout << " newBestUB=" << bestUB
                 << " lb=" << lb
                 << " ub=" << ub
                 << " stack=" << stack.size() << endl;
        }
        if (lb >= bestUB - 1e-9) {
            pruned++;
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " pruned lb=" << lb
                 << " stack=" << stack.size() << endl;
            continue;
        }

        int bestIdx = -1;
        int bestSpan = 0;
        for (int it = 0; it < N_it; ++it) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = CapIdx(it, t);
                int span = node.U[idx] - node.L[idx];
                if (span > bestSpan) {
                    bestSpan = span;
                    bestIdx = idx;
                }
            }
        }
        if (bestIdx < 0) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " leaf lb=" << lb
                 << " ub=" << ub
                 << " stack=" << stack.size() << endl;
            continue;
        }

        int l = node.L[bestIdx];
        int u = node.U[bestIdx];
        if (l >= u) {
            double globalLB = getGlobalLB(IloInfinity, false);
            printNodePrefix(globalLB);
            cout << " leaf lb=" << lb
                 << " ub=" << ub
                 << " stack=" << stack.size() << endl;
            continue;
        }
        int m = (l + u) / 2;
        int branchIt = bestIdx / (N_tp + 1);
        int branchT = bestIdx % (N_tp + 1);

        CapBBNode left = node;
        left.U[bestIdx] = m;
        CapBBNode right = node;
        right.L[bestIdx] = m + 1;
        auto pushChild = [&](CapBBNode& child) {
            if (!NormalizeNodeBounds(child.L, child.U)) {
                return;
            }
            if (SumU_EqCapAll(child.U) || SumL_EqCapAll(child.L)) {
                double val = PhiSumAtU(env, pbList, (SumU_EqCapAll(child.U) ? child.U : child.L), phiCache);
                if (val < bestUB) {
                    bestUB = val;
                }
                return;
            }
            child.lb = ComputeLB_DiscreteCap(env, pbList, child.L, child.U, phiCache);
            if (!std::isfinite(child.lb)) {
                return;
            }
            if (child.lb >= bestUB - 1e-9) {
                return;
            }
            stack.push_back(child);
        };
        pushChild(left);
        pushChild(right);

        double globalLB = getGlobalLB(IloInfinity, false);
        printNodePrefix(globalLB);
        cout << " branch it=" << branchIt
             << " t=" << branchT
             << " L=" << l
             << " U=" << u
             << " m=" << m
             << " lb=" << lb
             << " ub=" << ub
             << " stack=" << stack.size() << endl;

        if (globalLB >= bestUB - 1e-9) {
            break;
        }
    }

    if (optimalOut) {
        *optimalOut = true;
    }
    cout << "Capacity subproblem done tau=" << tauRep
         << " bestLB=" << getGlobalLB(IloInfinity, false)
         << " bestUB=" << bestUB
         << " nodes=" << iter
         << " pruned=" << pruned << endl;
    return bestUB;
}

static double SolveTauItemSubproblem_LP_Cap(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<double>& capIt,
    std::vector<double>& coefCap
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        coefCap.assign(N_tp + 1, 0.0);
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp+1, 0, 1, ILOFLOAT);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    IloRangeArray capCons(env);
    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= min(Cap, minBigM) * Y[t]);
        capCons.add(X[t] <= capIt[t]);
    }
    model.add(capCons);
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp+1) + t;
                int idxPrev = k * (N_tp+1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp+1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        coefCap.assign(N_tp + 1, 0.0);
        return IloInfinity;
    }

    IloNumArray dualCap(env);
    cplex.getDuals(dualCap, capCons);
    coefCap.assign(N_tp + 1, 0.0);
    for (int t = 1; t <= N_tp; ++t) {
        double pi = dualCap[t-1];
        coefCap[t] = pi;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauJointSubproblem(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    bool* optimalOut = nullptr
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        if (optimalOut) {
            *optimalOut = true;
        }
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp+1, 0, 1, ILOBOOL);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    int idxPrev = k * (N_tp+1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp+1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 900);
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauJointValidIneqCallback(Y, I, L, pbList, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        if (optimalOut) {
            *optimalOut = false;
        }
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    if (optimalOut) {
        IloAlgorithm::Status status = cplex.getStatus();
        *optimalOut = (status == IloAlgorithm::Optimal);
    }
    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauJointSubproblem_LP(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp+1, 0, 1, ILOFLOAT);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    int idxPrev = k * (N_tp+1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp+1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 300);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }
    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static RLObjective SolveTauMaCap(
    IloEnv env,
    const std::vector<int>& pbList,
    int maxIter
){
    RLObjective out{IloInfinity, IloInfinity};
    if (pbList.empty()) {
        out.lb = 0.0;
        out.ub = 0.0;
        return out;
    }

    IloModel master(env);
    IloCplex cplex(master);

    IloNumVarArray2 CapIT(env, N_it);
    IloNumVarArray ThetaI(env, N_it, 0.0, IloInfinity, ILOFLOAT);

    for (int it = 0; it < N_it; ++it) {
        CapIT[it] = IloNumVarArray(env, N_tp+1, 0.0, static_cast<IloNum>(Cap), ILOFLOAT);
        master.add(CapIT[it][0] == 0.0);
    }
    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capSum(env);
        for (int it = 0; it < N_it; ++it) {
            capSum += CapIT[it][t];
        }
        master.add(capSum == Cap);
        capSum.end();
    }

    IloExpr obj(env);
    for (int it = 0; it < N_it; ++it) {
        obj += ThetaI[it];
    }
    master.add(IloMinimize(env, obj));

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    double bestUB = IloInfinity;
    double bestLB = -IloInfinity;

    for (int iter = 0; iter < maxIter; ++iter) {
        if (!cplex.solve()) {
            break;
        }
        double lb = cplex.getObjValue();
        if (lb > bestLB) bestLB = lb;

        std::vector<std::vector<double>> capVals(N_it, std::vector<double>(N_tp + 1, 0.0));
        for (int it = 0; it < N_it; ++it) {
            for (int t = 1; t <= N_tp; ++t) {
                capVals[it][t] = cplex.getValue(CapIT[it][t]);
            }
        }

        for (int it = 0; it < N_it; ++it) {
            std::vector<double> coefCap(N_tp + 1, 0.0);
            double lpVal = SolveTauItemSubproblem_LP_Cap(env, it, pbList, capVals[it], coefCap);
            if (!std::isfinite(lpVal)) {
                continue;
            }
            double constLP = lpVal;
            for (int t = 1; t <= N_tp; ++t) {
                constLP -= coefCap[t] * capVals[it][t];
            }

            double mipVal = SolveTauItemSubproblem_Cap(env, it, pbList, capVals[it]);
            double constMIP = constLP;
            if (std::isfinite(mipVal)) {
                constMIP = mipVal;
                for (int t = 1; t <= N_tp; ++t) {
                    constMIP -= coefCap[t] * capVals[it][t];
                }
            }
            double rhs = (constMIP > constLP) ? constMIP : constLP;

            IloExpr cutExpr(env);
            cutExpr += rhs;
            for (int t = 1; t <= N_tp; ++t) {
                cutExpr += coefCap[t] * CapIT[it][t];
            }
            master.add(ThetaI[it] >= cutExpr);
            cutExpr.end();
        }

        double ub = 0.0;
        for (int it = 0; it < N_it; ++it) {
            double mipVal = SolveTauItemSubproblem_Cap(env, it, pbList, capVals[it]);
            if (std::isfinite(mipVal)) {
                ub += mipVal;
            } else {
                ub = IloInfinity;
                break;
            }
        }
        if (std::isfinite(ub) && ub < bestUB) {
            bestUB = ub;
        }
    }

    out.lb = std::isfinite(bestLB) ? bestLB : IloInfinity;
    out.ub = std::isfinite(bestUB) ? bestUB : IloInfinity;
    cplex.end();
    master.end();
    return out;
}

static RLObjective EvaluateF_ByTau_MaCap(
    IloEnv env,
    int zMask,
    int maxIter,
    std::unordered_map<int, RLObjective>& cacheMask,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            CacheMaskHit++;
            return itMask->second;
        }
    }

    double totalLB = 0.0;
    double totalUB = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return {IloInfinity, IloInfinity};
        }
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) continue;
        RLObjective val = SolveTauMaCap(env, pbList, maxIter);
        if (BBTimeLimitReached()) {
            return {IloInfinity, IloInfinity};
        }
        totalLB += val.lb;
        totalUB += val.ub;
    }

    RLObjective out{totalLB, totalUB};
    if (useCache) {
        cacheMask[zMask] = out;
    }
    return out;
}

static double SolveTauItemSubproblem_LR(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<double>& lambda,
    std::vector<int>& ySol,
    std::vector<double>& xSol,
    int zMask = -1
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return 0.0;
    }
    const IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp + 1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
        if (t < static_cast<int>(lambda.size())) {
            obj += lambda[t] * X[t];
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= std::min(Cap, minBigM) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                int idxPrev = k * (N_tp + 1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp + 1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);
    if (zMask >= 0 && (VI_pro1 == 1 || VI_pro2 == 1)) {
        cplex.use(new (env) TauItemValidIneqCallback(Y, I, L, pbList, item, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return IloInfinity;
    }

    ySol.assign(N_tp + 1, 0);
    xSol.assign(N_tp + 1, 0.0);
    for (int t = 0; t <= N_tp; ++t) {
        xSol[t] = cplex.getValue(X[t]);
        double yv = cplex.getValue(Y[t]);
        ySol[t] = (yv >= 0.5) ? 1 : 0;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauItemSubproblem_LR_ExcludePattern(
    IloEnv env,
    int item,
    const std::vector<int>& pbList,
    const std::vector<double>& lambda,
    const std::vector<int>& forbidPattern,
    std::vector<int>& ySol,
    std::vector<double>& xSol,
    double timeLimitSec
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return 0.0;
    }
    const IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray X(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    IloNumVarArray Y(env, N_tp + 1, 0, 1, ILOBOOL);

    IloNumVarArray2 I(env, N_sc);
    IloNumVarArray2 L(env, N_sc);
    for (int sc = 0; sc < N_sc; ++sc) {
        I[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        L[sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * X[t] + f[t][item] * Y[t]);
        if (t < static_cast<int>(lambda.size())) {
            obj += lambda[t] * X[t];
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                obj += rho_joint * (h[t][item] * I[sc][idx] + b[t][item] * L[sc][idx]);
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int t = 1; t <= N_tp; ++t) {
        int minBigM = Cap;
        for (int pb : pbList) {
            minBigM = std::min(minBigM, (int)BigM[item][pb][t]);
        }
        model.add(X[t] <= std::min(Cap, minBigM) * Y[t]);
    }
    model.add(X[0] == 0);
    model.add(Y[0] == 0);

    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            for (int t = 1; t <= N_tp; ++t) {
                int idx = k * (N_tp + 1) + t;
                int idxPrev = k * (N_tp + 1) + (t - 1);
                model.add(I[sc][idx] == I[sc][idxPrev] + X[t] - d[t][sc][pb][item] + L[sc][idx]);
            }
            int idx0 = k * (N_tp + 1);
            model.add(I[sc][idx0] == 0);
            model.add(L[sc][idx0] == 0);
        }
    }

    if ((int)forbidPattern.size() >= N_tp + 1) {
        IloExpr diffExpr(env);
        for (int t = 1; t <= N_tp; ++t) {
            if (forbidPattern[t] == 0) diffExpr += Y[t];
            else diffExpr += (1 - Y[t]);
        }
        model.add(diffExpr >= 1);
        diffExpr.end();
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, std::max(1.0, timeLimitSec));

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        ySol.assign(N_tp + 1, 0);
        xSol.assign(N_tp + 1, 0.0);
        return IloInfinity;
    }

    ySol.assign(N_tp + 1, 0);
    xSol.assign(N_tp + 1, 0.0);
    for (int t = 0; t <= N_tp; ++t) {
        xSol[t] = cplex.getValue(X[t]);
        double yv = cplex.getValue(Y[t]);
        ySol[t] = (yv >= 0.5) ? 1 : 0;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauJointFixedY_LP(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<std::vector<int>>& yFix
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * yFix[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= std::min(Cap, minBigM) * yFix[it][t]);
        }
        model.add(X[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    int idxPrev = k * (N_tp + 1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp + 1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double SolveTauJointFixedY_LP_WithSol(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<std::vector<int>>& yFix,
    std::vector<std::vector<double>>& xSol
){
    const int pbCount = static_cast<int>(pbList.size());
    xSol.assign(N_it, std::vector<double>(N_tp + 1, 0.0));
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * yFix[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= std::min(Cap, minBigM) * yFix[it][t]);
        }
        model.add(X[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    int idxPrev = k * (N_tp + 1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp + 1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    for (int it = 0; it < N_it; ++it) {
        for (int t = 0; t <= N_tp; ++t) {
            xSol[it][t] = cplex.getValue(X[it][t]);
        }
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static void AddTauJointPathCutsSuffix(
    IloEnv env,
    IloModel& model,
    const std::vector<int>& pbList,
    const IloNumVarArray2& Y,
    const IloNumVarArray3& I,
    const IloNumVarArray3& L
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0 || N_tp <= 1) {
        return;
    }
    const double capD = static_cast<double>(Cap);

    for (int it = 0; it < N_it; ++it) {
        for (int pbIdx = 0; pbIdx < pbCount; ++pbIdx) {
            int pb = pbList[pbIdx];
            for (int sc = 0; sc < N_sc; ++sc) {
                std::vector<double> suffixDemand(N_tp + 2, 0.0);
                for (int t = N_tp; t >= 1; --t) {
                    suffixDemand[t] = suffixDemand[t + 1] + d[t][sc][pb][it];
                }
                for (int k = 1; k < N_tp; ++k) {
                    IloExpr cutExpr(env);
                    int idxK = pbIdx * (N_tp + 1) + k;
                    cutExpr += I[it][sc][idxK];
                    for (int v = k + 1; v <= N_tp; ++v) {
                        int idxV = pbIdx * (N_tp + 1) + v;
                        cutExpr += L[it][sc][idxV];
                        cutExpr += -d[v][sc][pb][it];
                    }
                    for (int l = k + 1; l <= N_tp; ++l) {
                        double coeff = std::min(capD, suffixDemand[l]);
                        if (coeff > 1e-9) {
                            cutExpr += coeff * Y[it][l];
                        }
                    }
                    model.add(cutExpr >= 0);
                    cutExpr.end();
                    VIPathCount++;
                }
            }
        }
    }
}

static double SolveTauJointRelaxedY_StrengthenedLP_WithSol(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<double>& itemObjLB,
    std::vector<std::vector<double>>& ySol,
    std::vector<std::vector<double>>& xSol
){
    const int pbCount = static_cast<int>(pbList.size());
    ySol.assign(N_it, std::vector<double>(N_tp + 1, 0.0));
    xSol.assign(N_it, std::vector<double>(N_tp + 1, 0.0));
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp + 1, 0, 1, ILOFLOAT); // relaxed setup
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= std::min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    int idxPrev = k * (N_tp + 1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp + 1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    // Strengthening: static path inequalities for relaxed-Y LP.
    if (VI_pro1 == 1) {
        AddTauJointPathCutsSuffix(env, model, pbList, Y, I, L);
    }

    // Strengthening: each item objective cannot go below uncoupled item-wise optimum.
    if ((int)itemObjLB.size() == N_it) {
        for (int it = 0; it < N_it; ++it) {
            if (!std::isfinite(itemObjLB[it])) continue;
            IloExpr objItem(env);
            for (int t = 1; t <= N_tp; ++t) {
                objItem += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
            }
            for (int k = 0; k < pbCount; ++k) {
                int pb = pbList[k];
                for (int sc = 0; sc < N_sc; ++sc) {
                    for (int t = 1; t <= N_tp; ++t) {
                        int idx = k * (N_tp + 1) + t;
                        objItem += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                    }
                }
            }
            model.add(objItem >= itemObjLB[it] - 1e-6);
            objItem.end();
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 120);

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    for (int it = 0; it < N_it; ++it) {
        for (int t = 0; t <= N_tp; ++t) {
            xSol[it][t] = cplex.getValue(X[it][t]);
            ySol[it][t] = cplex.getValue(Y[it][t]);
        }
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static TauRelaxedLPInfo SolveTauJointRelaxedY_LPWithDuals(
    IloEnv env,
    const std::vector<int>& pbList
) {
    TauRelaxedLPInfo out;
    out.lambda.assign(N_tp + 1, 0.0);
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        out.lpLB = 0.0;
        out.valid = true;
        return out;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp + 1, 0, 1, ILOFLOAT);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= std::min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    IloRangeArray capCons(env);
    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        capCons.add(capExpr <= Cap);
        capExpr.end();
    }
    model.add(capCons);

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    int idxPrev = k * (N_tp + 1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp + 1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, std::max(1.0, SetupCertItemTiLim));

    if (cplex.solve()) {
        out.lpLB = cplex.getObjValue();
        for (int t = 1; t <= N_tp; ++t) {
            out.lambda[t] = std::max(0.0, cplex.getDual(capCons[t - 1]));
        }
        out.valid = true;
    }

    obj.end();
    cplex.end();
    model.end();

    return out;
}

static double SolveTauJointPartialY_MIP(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    const std::vector<std::vector<int>>& yFix,
    const std::vector<char>& freeT,
    double timeLimitSec,
    std::vector<std::vector<int>>* yBestOut = nullptr,
    const std::vector<std::vector<char>>* freeYComp = nullptr
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp+1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp+1, 0, 1, ILOBOOL);
    }

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            bool isFree = false;
            if (freeYComp && it < static_cast<int>(freeYComp->size()) &&
                t < static_cast<int>((*freeYComp)[it].size())) {
                isFree = ((*freeYComp)[it][t] != 0);
            } else if (t < static_cast<int>(freeT.size())) {
                isFree = (freeT[t] != 0);
            }
            if (!isFree) {
                int val = (it < static_cast<int>(yFix.size()) && t < static_cast<int>(yFix[it].size())) ? yFix[it][t] : 0;
                Y[it][t].setBounds(val, val);
            }
        }
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp+1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp+1) + t;
                    int idxPrev = k * (N_tp+1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp+1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, std::max(1.0, timeLimitSec));
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauJointValidIneqCallback(Y, I, L, pbList, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (!cplex.solve()) {
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    if (yBestOut) {
        yBestOut->assign(N_it, std::vector<int>(N_tp + 1, 0));
        for (int it = 0; it < N_it; ++it) {
            for (int t = 0; t <= N_tp; ++t) {
                double yv = cplex.getValue(Y[it][t]);
                (*yBestOut)[it][t] = (yv >= 0.5) ? 1 : 0;
            }
        }
    }

    double objVal = cplex.getObjValue();
    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static double EvaluateTauItemPlanCost(
    const std::vector<int>& pbList,
    int item,
    const std::vector<double>& xItem,
    const std::vector<int>& yItem
) {
    const int pbCount = static_cast<int>(pbList.size());
    double rhoJoint = 1.0 / (static_cast<double>(N_pb) * static_cast<double>(N_sc));
    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    double obj = 0.0;
    for (int t = 1; t <= N_tp; ++t) {
        obj += pbWeight * (p[t][item] * xItem[t] + f[t][item] * static_cast<double>(yItem[t]));
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int sc = 0; sc < N_sc; ++sc) {
            double inv = 0.0;
            for (int t = 1; t <= N_tp; ++t) {
                double net = inv + xItem[t] - d[t][sc][pb][item];
                if (net >= 0.0) {
                    inv = net;
                    obj += rhoJoint * h[t][item] * inv;
                } else {
                    inv = 0.0;
                    obj += rhoJoint * b[t][item] * (-net);
                }
            }
        }
    }
    return obj;
}

static bool BuildTauHeuristicSeedFromLambda(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    const std::vector<double>& lambda,
    EvalStats* stats,
    bool useActualAsItemLb,
    TauHeuristicSeed& seed
) {
    seed = TauHeuristicSeed();
    seed.y.assign(N_it, std::vector<int>(N_tp + 1, 0));
    seed.x.assign(N_it, std::vector<double>(N_tp + 1, 0.0));
    seed.itemObjLB.assign(N_it, IloInfinity);
    seed.sumX.assign(N_tp + 1, 0.0);
    seed.actualObj = 0.0;
    seed.lagObj = 0.0;

    for (int it = 0; it < N_it; ++it) {
        std::vector<int> ySol;
        std::vector<double> xSol;
        auto t0 = std::chrono::steady_clock::now();
        double lagObj = SolveTauItemSubproblem_LR(env, it, pbList, lambda, ySol, xSol, zMask);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stats) {
            stats->itemSolved += 1;
            stats->timeItem += dt;
        }
        if (!std::isfinite(lagObj)) {
            return false;
        }
        double actualObj = EvaluateTauItemPlanCost(pbList, it, xSol, ySol);
        if (!std::isfinite(actualObj)) {
            return false;
        }
        seed.lagObj += lagObj;
        seed.actualObj += actualObj;
        if (useActualAsItemLb) {
            seed.itemObjLB[it] = actualObj;
        }
        for (int t = 1; t <= N_tp; ++t) {
            seed.sumX[t] += xSol[t];
            seed.x[it][t] = xSol[t];
            seed.y[it][t] = ySol[t];
        }
    }

    seed.valid = true;
    return true;
}

static double SolveTauJointHeuristic(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    EvalStats* stats,
    std::vector<std::vector<int>>* yOut = nullptr,
    const TauHeuristicSeed* seedIn = nullptr
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        if (yOut) {
            yOut->assign(N_it, std::vector<int>(N_tp + 1, 0));
        }
        return 0.0;
    }
    const double eps = 1e-6;
    auto storeY = [&](const std::vector<std::vector<int>>& yCand) {
        if (yOut) {
            *yOut = yCand;
        }
    };

    auto evalItemCost = [&](int item,
                            const std::vector<double>& xItem,
                            const std::vector<int>& yItem) -> double {
        return EvaluateTauItemPlanCost(pbList, item, xItem, yItem);
    };

    auto cleanupPlan = [&](std::vector<std::vector<double>>& xPlan,
                           std::vector<std::vector<int>>& yPlan) {
        for (int it = 0; it < N_it; ++it) {
            xPlan[it][0] = 0.0;
            yPlan[it][0] = 0;
            for (int t = 1; t <= N_tp; ++t) {
                if (xPlan[it][t] <= eps) {
                    xPlan[it][t] = 0.0;
                    yPlan[it][t] = 0;
                } else {
                    yPlan[it][t] = 1;
                }
            }
        }
    };

    auto repairPlan = [&](std::vector<std::vector<double>>& xPlan,
                          std::vector<std::vector<int>>& yPlan,
                          std::vector<double>& sumXPlan,
                          double& repairedObj) -> bool {
        struct MoveChoice {
            bool valid = false;
            int item = -1;
            int toT = -1; // -1 means cut without move
            double q = 0.0;
            double delta = IloInfinity;
            double newCost = IloInfinity;
        };

        std::vector<double> itemCost(N_it, 0.0);
        for (int it = 0; it < N_it; ++it) {
            itemCost[it] = evalItemCost(it, xPlan[it], yPlan[it]);
            if (!std::isfinite(itemCost[it])) return false;
        }

        for (int t = 1; t <= N_tp; ++t) {
            int guard = 0;
            while (sumXPlan[t] > static_cast<double>(Cap) + eps) {
                if (++guard > 20000) {
                    return false;
                }
                double overload = sumXPlan[t] - static_cast<double>(Cap);
                MoveChoice best;

                auto betterChoice = [&](double delta, int toT, double q) -> bool {
                    if (!best.valid) return true;
                    if (delta < best.delta - 1e-9) return true;
                    if (std::fabs(delta - best.delta) <= 1e-9) {
                        bool moveNow = (toT >= 1);
                        bool moveBest = (best.toT >= 1);
                        if (moveNow != moveBest) return moveNow;
                        if (q > best.q + 1e-9) return true;
                    }
                    return false;
                };

                for (int it = 0; it < N_it; ++it) {
                    double avail = xPlan[it][t];
                    if (avail <= eps) continue;

                    double qBase = std::min(overload, avail);
                    if (qBase <= eps) continue;

                    {
                        std::vector<double> xNew = xPlan[it];
                        std::vector<int> yNew = yPlan[it];
                        xNew[t] -= qBase;
                        if (xNew[t] <= eps) {
                            xNew[t] = 0.0;
                            yNew[t] = 0;
                        }
                        double newCost = evalItemCost(it, xNew, yNew);
                        double delta = newCost - itemCost[it];
                        if (betterChoice(delta, -1, qBase)) {
                            best.valid = true;
                            best.item = it;
                            best.toT = -1;
                            best.q = qBase;
                            best.delta = delta;
                            best.newCost = newCost;
                        }
                    }

                    for (int t2 = 1; t2 <= N_tp; ++t2) {
                        if (t2 == t) continue;
                        double slack = static_cast<double>(Cap) - sumXPlan[t2];
                        if (slack <= eps) continue;
                        double q = std::min(qBase, slack);
                        if (q <= eps) continue;

                        std::vector<double> xNew = xPlan[it];
                        std::vector<int> yNew = yPlan[it];
                        xNew[t] -= q;
                        if (xNew[t] <= eps) {
                            xNew[t] = 0.0;
                            yNew[t] = 0;
                        }
                        xNew[t2] += q;
                        if (xNew[t2] > eps) {
                            yNew[t2] = 1;
                        }
                        double newCost = evalItemCost(it, xNew, yNew);
                        double delta = newCost - itemCost[it];
                        if (betterChoice(delta, t2, q)) {
                            best.valid = true;
                            best.item = it;
                            best.toT = t2;
                            best.q = q;
                            best.delta = delta;
                            best.newCost = newCost;
                        }
                    }
                }

                if (!best.valid) return false;

                int it = best.item;
                xPlan[it][t] -= best.q;
                if (xPlan[it][t] <= eps) {
                    xPlan[it][t] = 0.0;
                    yPlan[it][t] = 0;
                }
                sumXPlan[t] -= best.q;
                if (best.toT >= 1) {
                    xPlan[it][best.toT] += best.q;
                    yPlan[it][best.toT] = 1;
                    sumXPlan[best.toT] += best.q;
                }
                itemCost[it] = best.newCost;
            }
        }

        cleanupPlan(xPlan, yPlan);
        std::fill(sumXPlan.begin(), sumXPlan.end(), 0.0);
        repairedObj = 0.0;
        for (int it = 0; it < N_it; ++it) {
            for (int t = 1; t <= N_tp; ++t) {
                sumXPlan[t] += xPlan[it][t];
            }
            double c = evalItemCost(it, xPlan[it], yPlan[it]);
            if (!std::isfinite(c)) return false;
            repairedObj += c;
        }
        for (int t = 1; t <= N_tp; ++t) {
            if (sumXPlan[t] > static_cast<double>(Cap) + 1e-5) return false;
        }
        return true;
    };

    TauHeuristicSeed seed;
    if (seedIn && seedIn->valid) {
        seed = *seedIn;
    } else {
        std::vector<double> lambda0(N_tp + 1, 0.0);
        if (!BuildTauHeuristicSeedFromLambda(env, pbList, zMask, lambda0, stats, true, seed)) {
            return IloInfinity;
        }
    }

    std::vector<std::vector<int>> yFix = seed.y;
    std::vector<std::vector<double>> xFix = seed.x;
    std::vector<double> itemObjLB = seed.itemObjLB;
    std::vector<double> sumX = seed.sumX;
    double sumObj = seed.actualObj;

    std::vector<char> freeT(N_tp + 1, 0);
    bool hasViolation = false;
    int radius = (HeurFreeWindow <= 0) ? 0 : HeurFreeWindow;
    for (int t = 1; t <= N_tp; ++t) {
        if (sumX[t] > static_cast<double>(Cap) + eps) {
            hasViolation = true;
            for (int dt = -radius; dt <= radius; ++dt) {
                int tt = t + dt;
                if (tt >= 1 && tt <= N_tp) {
                    freeT[tt] = 1;
                }
            }
        }
    }

    if (!hasViolation) {
        storeY(yFix);
        return sumObj;
    }

    if (HeurRepairMode == 3) {
        std::vector<std::vector<int>> yCurr = yFix;
        std::vector<std::vector<double>> xJoint;
        double bestUB = IloInfinity;
        bool v3Ok = true;

        auto solveJointWithY = [&](const std::vector<std::vector<int>>& yIn,
                                   std::vector<std::vector<double>>& xOut,
                                   double& objOut) -> bool {
            auto t0 = std::chrono::steady_clock::now();
            objOut = SolveTauJointFixedY_LP_WithSol(env, pbList, yIn, xOut);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            return std::isfinite(objOut);
        };

        double jointObj = IloInfinity;
        if (!solveJointWithY(yCurr, xJoint, jointObj)) {
            v3Ok = false;
        } else {
            bestUB = jointObj;
        }

        bool converged = false;
        int maxIter = std::max(1, HeurRepairV3MaxIter);
        for (int iter = 0; v3Ok && iter < maxIter; ++iter) {
            std::vector<std::vector<int>> yNext = yCurr;
            std::vector<std::vector<double>> xNext = xJoint;

            if (HeurRepairV3UpdateMode == 0) {
                std::vector<double> sumXJoint(N_tp + 1, 0.0);
                for (int t = 1; t <= N_tp; ++t) {
                    for (int j = 0; j < N_it; ++j) {
                        sumXJoint[t] += xJoint[j][t];
                    }
                }

                std::vector<double> objByItem(N_it, IloInfinity);
                std::vector<double> timeByItem(N_it, 0.0);
                std::vector<char> okByItem(N_it, 0);
                std::vector<std::thread> workers;
                workers.reserve(N_it);

                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double resid = static_cast<double>(Cap) - (sumXJoint[t] - xJoint[it][t]);
                        capIt[t] = std::max(0.0, resid);
                    }
                    workers.emplace_back([&, it, capIt]() {
                        try {
                            IloEnv envLocal;
                            std::vector<int> yLoc;
                            std::vector<double> xLoc;
                            auto t0 = std::chrono::steady_clock::now();
                            double obj = SolveTauItemSubproblem_Cap_WithSol(
                                envLocal, it, pbList, capIt, zMask, yLoc, xLoc, 1);
                            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                            envLocal.end();
                            if (std::isfinite(obj)) {
                                yNext[it] = yLoc;
                                xNext[it] = xLoc;
                                objByItem[it] = obj;
                                timeByItem[it] = dt;
                                okByItem[it] = 1;
                            }
                        } catch (...) {
                            okByItem[it] = 0;
                        }
                    });
                }
                for (size_t k = 0; k < workers.size(); ++k) {
                    workers[k].join();
                }
                for (int it = 0; it < N_it; ++it) {
                    if (okByItem[it] == 0) {
                        v3Ok = false;
                        break;
                    }
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += timeByItem[it];
                    }
                }
            } else {
                std::vector<std::vector<double>> xRef = xJoint;
                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double others = 0.0;
                        for (int j = 0; j < N_it; ++j) {
                            if (j == it) continue;
                            others += xRef[j][t];
                        }
                        capIt[t] = std::max(0.0, static_cast<double>(Cap) - others);
                    }
                    std::vector<int> yLoc;
                    std::vector<double> xLoc;
                    auto t0 = std::chrono::steady_clock::now();
                    double obj = SolveTauItemSubproblem_Cap_WithSol(
                        env, it, pbList, capIt, zMask, yLoc, xLoc, 2);
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += dt;
                    }
                    if (!std::isfinite(obj)) {
                        v3Ok = false;
                        break;
                    }
                    yNext[it] = yLoc;
                    xNext[it] = xLoc;
                    xRef[it] = xLoc;
                }
            }

            if (!v3Ok) break;

            bool changed = false;
            for (int it = 0; it < N_it && !changed; ++it) {
                for (int t = 1; t <= N_tp; ++t) {
                    if (yNext[it][t] != yCurr[it][t]) {
                        changed = true;
                        break;
                    }
                }
            }

            if (!changed) {
                converged = true;
                break;
            }

            yCurr.swap(yNext);
            if (!solveJointWithY(yCurr, xJoint, jointObj)) {
                v3Ok = false;
                break;
            }
            if (jointObj < bestUB) bestUB = jointObj;
        }

        if (v3Ok && !converged) {
            // Ensure current Y has an evaluated joint LP value when max-iter is reached.
            double objFinal = IloInfinity;
            if (solveJointWithY(yCurr, xJoint, objFinal)) {
                if (objFinal < bestUB) bestUB = objFinal;
            } else {
                v3Ok = false;
            }
        }

        if (v3Ok && std::isfinite(bestUB)) {
            storeY(yCurr);
            return bestUB;
        }
        // Fallback to legacy heuristic if V3 alternating fails.
    }

    if (HeurRepairMode == 4) {
        std::vector<std::vector<int>> yCurr = yFix;
        std::vector<std::vector<double>> xRef;
        bool v4Ok = true;

        auto solveRelaxed = [&](std::vector<std::vector<double>>& xOut,
                                double& objOut) -> bool {
            std::vector<std::vector<double>> yRelaxTmp;
            auto t0 = std::chrono::steady_clock::now();
            objOut = SolveTauJointRelaxedY_StrengthenedLP_WithSol(env, pbList, itemObjLB, yRelaxTmp, xOut);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            return std::isfinite(objOut);
        };

        auto solveFinalWithY = [&](const std::vector<std::vector<int>>& yIn,
                                   double& objOut) -> bool {
            auto t0 = std::chrono::steady_clock::now();
            objOut = SolveTauJointFixedY_LP(env, pbList, yIn);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            return std::isfinite(objOut);
        };

        double relaxObj = IloInfinity;
        if (!solveRelaxed(xRef, relaxObj)) {
            v4Ok = false;
        }

        bool converged = false;
        int maxIter = std::max(1, HeurRepairV3MaxIter);
        for (int iter = 0; v4Ok && iter < maxIter; ++iter) {
            std::vector<std::vector<int>> yNext = yCurr;

            if (HeurRepairV3UpdateMode == 0) {
                std::vector<double> sumXRef(N_tp + 1, 0.0);
                for (int t = 1; t <= N_tp; ++t) {
                    for (int j = 0; j < N_it; ++j) {
                        sumXRef[t] += xRef[j][t];
                    }
                }

                std::vector<double> timeByItem(N_it, 0.0);
                std::vector<char> okByItem(N_it, 0);
                std::vector<std::thread> workers;
                workers.reserve(N_it);

                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double resid = static_cast<double>(Cap) - (sumXRef[t] - xRef[it][t]);
                        capIt[t] = std::max(0.0, resid);
                    }
                    workers.emplace_back([&, it, capIt]() {
                        try {
                            IloEnv envLocal;
                            std::vector<int> yLoc;
                            std::vector<double> xLoc;
                            auto t0 = std::chrono::steady_clock::now();
                            double obj = SolveTauItemSubproblem_Cap_WithSol(
                                envLocal, it, pbList, capIt, zMask, yLoc, xLoc, 1);
                            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                            envLocal.end();
                            if (std::isfinite(obj)) {
                                yNext[it] = yLoc;
                                timeByItem[it] = dt;
                                okByItem[it] = 1;
                            }
                        } catch (...) {
                            okByItem[it] = 0;
                        }
                    });
                }
                for (size_t k = 0; k < workers.size(); ++k) {
                    workers[k].join();
                }
                for (int it = 0; it < N_it; ++it) {
                    if (okByItem[it] == 0) {
                        v4Ok = false;
                        break;
                    }
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += timeByItem[it];
                    }
                }
            } else {
                std::vector<std::vector<double>> xTmp = xRef;
                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double others = 0.0;
                        for (int j = 0; j < N_it; ++j) {
                            if (j == it) continue;
                            others += xTmp[j][t];
                        }
                        capIt[t] = std::max(0.0, static_cast<double>(Cap) - others);
                    }
                    std::vector<int> yLoc;
                    std::vector<double> xLoc;
                    auto t0 = std::chrono::steady_clock::now();
                    double obj = SolveTauItemSubproblem_Cap_WithSol(
                        env, it, pbList, capIt, zMask, yLoc, xLoc, 2);
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += dt;
                    }
                    if (!std::isfinite(obj)) {
                        v4Ok = false;
                        break;
                    }
                    yNext[it] = yLoc;
                    xTmp[it] = xLoc;
                }
            }

            if (!v4Ok) break;

            bool changed = false;
            for (int it = 0; it < N_it && !changed; ++it) {
                for (int t = 1; t <= N_tp; ++t) {
                    if (yNext[it][t] != yCurr[it][t]) {
                        changed = true;
                        break;
                    }
                }
            }

            yCurr.swap(yNext);
            if (!changed) {
                break;
            }

            if (!solveRelaxed(xRef, relaxObj)) {
                v4Ok = false;
                break;
            }
        }

        if (v4Ok) {
            double ubFinal = IloInfinity;
            if (solveFinalWithY(yCurr, ubFinal)) {
                storeY(yCurr);
                return ubFinal;
            }
            v4Ok = false;
        }
        // Fallback to legacy heuristic if V4 alternating fails.
    }

    if (HeurRepairMode == 5) {
        std::vector<std::vector<int>> yCurr = yFix;
        std::vector<std::vector<double>> xJoint;
        std::vector<int> flipByT(N_tp + 1, 0);
        double bestUB = IloInfinity;
        bool v5Ok = true;

        auto solveJointWithY = [&](const std::vector<std::vector<int>>& yIn,
                                   std::vector<std::vector<double>>& xOut,
                                   double& objOut) -> bool {
            auto t0 = std::chrono::steady_clock::now();
            objOut = SolveTauJointFixedY_LP_WithSol(env, pbList, yIn, xOut);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            return std::isfinite(objOut);
        };

        double jointObj = IloInfinity;
        if (!solveJointWithY(yCurr, xJoint, jointObj)) {
            v5Ok = false;
        } else {
            bestUB = jointObj;
        }

        int maxIter = std::max(1, HeurRepairV3MaxIter);
        for (int iter = 0; v5Ok && iter < maxIter; ++iter) {
            std::vector<std::vector<int>> yNext = yCurr;
            std::vector<std::vector<double>> xNext = xJoint;

            if (HeurRepairV3UpdateMode == 0) {
                std::vector<double> sumXJoint(N_tp + 1, 0.0);
                for (int t = 1; t <= N_tp; ++t) {
                    for (int j = 0; j < N_it; ++j) {
                        sumXJoint[t] += xJoint[j][t];
                    }
                }

                std::vector<double> timeByItem(N_it, 0.0);
                std::vector<char> okByItem(N_it, 0);
                std::vector<std::thread> workers;
                workers.reserve(N_it);

                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double resid = static_cast<double>(Cap) - (sumXJoint[t] - xJoint[it][t]);
                        capIt[t] = std::max(0.0, resid);
                    }
                    workers.emplace_back([&, it, capIt]() {
                        try {
                            IloEnv envLocal;
                            std::vector<int> yLoc;
                            std::vector<double> xLoc;
                            auto t0 = std::chrono::steady_clock::now();
                            double obj = SolveTauItemSubproblem_Cap_WithSol(
                                envLocal, it, pbList, capIt, zMask, yLoc, xLoc, 1);
                            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                            envLocal.end();
                            if (std::isfinite(obj)) {
                                yNext[it] = yLoc;
                                xNext[it] = xLoc;
                                timeByItem[it] = dt;
                                okByItem[it] = 1;
                            }
                        } catch (...) {
                            okByItem[it] = 0;
                        }
                    });
                }
                for (size_t k = 0; k < workers.size(); ++k) {
                    workers[k].join();
                }
                for (int it = 0; it < N_it; ++it) {
                    if (okByItem[it] == 0) {
                        v5Ok = false;
                        break;
                    }
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += timeByItem[it];
                    }
                }
            } else {
                std::vector<std::vector<double>> xRef = xJoint;
                for (int it = 0; it < N_it; ++it) {
                    std::vector<double> capIt(N_tp + 1, 0.0);
                    for (int t = 1; t <= N_tp; ++t) {
                        double others = 0.0;
                        for (int j = 0; j < N_it; ++j) {
                            if (j == it) continue;
                            others += xRef[j][t];
                        }
                        capIt[t] = std::max(0.0, static_cast<double>(Cap) - others);
                    }
                    std::vector<int> yLoc;
                    std::vector<double> xLoc;
                    auto t0 = std::chrono::steady_clock::now();
                    double obj = SolveTauItemSubproblem_Cap_WithSol(
                        env, it, pbList, capIt, zMask, yLoc, xLoc, 2);
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (stats) {
                        stats->itemSolved += 1;
                        stats->timeItem += dt;
                    }
                    if (!std::isfinite(obj)) {
                        v5Ok = false;
                        break;
                    }
                    yNext[it] = yLoc;
                    xNext[it] = xLoc;
                    xRef[it] = xLoc;
                }
            }

            if (!v5Ok) break;

            bool changed = false;
            for (int it = 0; it < N_it; ++it) {
                for (int t = 1; t <= N_tp; ++t) {
                    if (yNext[it][t] != yCurr[it][t]) {
                        changed = true;
                        flipByT[t] += 1;
                    }
                }
            }
            if (!changed) {
                break;
            }

            yCurr.swap(yNext);
            if (!solveJointWithY(yCurr, xJoint, jointObj)) {
                v5Ok = false;
                break;
            }
            if (jointObj < bestUB) bestUB = jointObj;
        }

        if (v5Ok && std::isfinite(bestUB)) {
            const double capD = std::max(1.0, static_cast<double>(Cap));
            std::vector<double> score(N_tp + 1, 0.0);
            std::vector<char> critical(N_tp + 1, 0);

            for (int t = 1; t <= N_tp; ++t) {
                double overload0 = std::max(0.0, sumX[t] - static_cast<double>(Cap)) / capD;
                if (overload0 > 1e-9) {
                    critical[t] = 1;
                }
                double sumXJoint_t = 0.0;
                for (int it = 0; it < N_it; ++it) {
                    sumXJoint_t += xJoint[it][t];
                }
                double util = sumXJoint_t / capD;

                double demAvg = 0.0;
                for (int k = 0; k < pbCount; ++k) {
                    int pb = pbList[k];
                    for (int sc = 0; sc < N_sc; ++sc) {
                        for (int it = 0; it < N_it; ++it) {
                            demAvg += d[t][sc][pb][it];
                        }
                    }
                }
                demAvg /= std::max(1, pbCount * N_sc);
                double demandPressure = demAvg / capD;

                double flipRatio = static_cast<double>(flipByT[t]) / std::max(1, N_it);
                score[t] = 5.0 * overload0 + 2.0 * util + demandPressure + flipRatio;
            }

            std::vector<std::pair<double, int>> rank;
            rank.reserve(N_tp);
            for (int t = 1; t <= N_tp; ++t) {
                rank.push_back(std::make_pair(score[t], t));
            }
            std::sort(rank.begin(), rank.end(), [](const std::pair<double, int>& a,
                                                   const std::pair<double, int>& b) {
                if (a.first == b.first) return a.second < b.second;
                return a.first > b.first;
            });
            int topK = std::max(1, static_cast<int>(std::ceil(0.2 * static_cast<double>(N_tp))));
            for (int k = 0; k < topK && k < static_cast<int>(rank.size()); ++k) {
                critical[rank[k].second] = 1;
            }

            std::vector<char> freeTLocal(N_tp + 1, 0);
            int localRadius = std::max(0, HeurFreeWindow);
            for (int t = 1; t <= N_tp; ++t) {
                if (!critical[t]) continue;
                for (int dt = -localRadius; dt <= localRadius; ++dt) {
                    int tt = t + dt;
                    if (tt >= 1 && tt <= N_tp) {
                        freeTLocal[tt] = 1;
                    }
                }
            }
            bool anyFree = false;
            for (int t = 1; t <= N_tp; ++t) {
                if (freeTLocal[t]) {
                    anyFree = true;
                    break;
                }
            }
            if (!anyFree && !rank.empty()) {
                freeTLocal[rank[0].second] = 1;
            }

            const double miniMipTiLim = 10.0;
            auto t0 = std::chrono::steady_clock::now();
            double ubInt = SolveTauJointPartialY_MIP(env, pbList, zMask, yCurr, freeTLocal, miniMipTiLim);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            if (std::isfinite(ubInt) && ubInt < bestUB) {
                bestUB = ubInt;
            }
            storeY(yCurr);
            return bestUB;
        }
        // Fallback to legacy heuristic if V5 hybrid fails.
    }

    if (HeurRepairMode == 1 || HeurRepairMode == 2) {
        std::vector<std::vector<double>> xRepair = xFix;
        std::vector<std::vector<int>> yRepair = yFix;
        std::vector<double> sumXRepair = sumX;
        double ubRepair = IloInfinity;
        bool repaired = repairPlan(xRepair, yRepair, sumXRepair, ubRepair);
        if (repaired && std::isfinite(ubRepair)) {
            double ubBest = ubRepair;
            if (HeurRepairMode == 2) {
                auto t0 = std::chrono::steady_clock::now();
                double ubLP = SolveTauJointFixedY_LP(env, pbList, yRepair);
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->jointSolved += 1;
                    stats->timeJoint += dt;
                }
                if (std::isfinite(ubLP) && ubLP < ubBest) {
                    ubBest = ubLP;
                }
            }
            storeY(yRepair);
            return ubBest;
        }
    }

    double ub = IloInfinity;
    if (HeurFreeWindow == 0) {
        auto t0 = std::chrono::steady_clock::now();
        ub = SolveTauJointFixedY_LP(env, pbList, yFix);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stats) {
            stats->jointSolved += 1;
            stats->timeJoint += dt;
        }
    } else {
        const double miniMipTiLim = 10.0;
        auto t0 = std::chrono::steady_clock::now();
        ub = SolveTauJointPartialY_MIP(env, pbList, zMask, yFix, freeT, miniMipTiLim);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stats) {
            stats->jointSolved += 1;
            stats->timeJoint += dt;
        }
        if (!std::isfinite(ub)) {
            t0 = std::chrono::steady_clock::now();
            ub = SolveTauJointFixedY_LP(env, pbList, yFix);
            dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
        }
    }
    storeY(yFix);
    return ub;
}

static bool IsValidYPlan(const std::vector<std::vector<int>>& yPlan) {
    if ((int)yPlan.size() != N_it) {
        return false;
    }
    for (int it = 0; it < N_it; ++it) {
        if ((int)yPlan[it].size() < N_tp + 1) {
            return false;
        }
    }
    return true;
}

static std::string YPlanSignature(const std::vector<std::vector<int>>& yPlan) {
    std::string key;
    if (!IsValidYPlan(yPlan)) {
        return key;
    }
    key.reserve(N_it * (N_tp + 2));
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            key.push_back(yPlan[it][t] ? '1' : '0');
        }
        key.push_back('|');
    }
    return key;
}

static bool ShouldRecertifyApproxIncumbent(
    double prevUb,
    double newUb,
    double globalLb
) {
    if (!std::isfinite(prevUb) || !std::isfinite(newUb) || !std::isfinite(globalLb)) {
        return false;
    }
    const double kMinRelImprove = 0.01;
    const double kNearFactor = 1.5;
    const double denom = std::max(1.0, std::fabs(prevUb));
    const double relImprove = (prevUb - newUb) / denom;
    if (relImprove + 1e-12 < kMinRelImprove) {
        return false;
    }
    const double rhs = (1.0 + kNearFactor * std::max(0.0, SetupCertTau)) * globalLb;
    const double eps = 1e-6 * std::max(1.0, std::fabs(rhs));
    return newUb <= rhs + eps;
}

static inline bool IsNAReuseWht() {
    return (ActiveWht == 27 || ActiveWht == 28 || ActiveWht == 29);
}

static inline bool IsCertifiedHeurWht() {
    return (ActiveWht == 28);
}

static inline bool IsApproxCertWht() {
    return (ActiveWht == 29);
}

static inline bool IsAIRWht() {
    return (ActiveWht == 30);
}

static inline bool SetupCertActive() {
    return (SetupCertEnable != 0 && (IsCertifiedHeurWht() || IsApproxCertWht()));
}

static inline double ApproxGapMetric(double lb, double ub) {
    if (!std::isfinite(lb) || !std::isfinite(ub)) {
        return IloInfinity;
    }
    const double denom = std::max(1.0, std::fabs(lb));
    return (ub - lb) / denom;
}

static inline bool AcceptApproxGap(double lb, double ub) {
    return (ApproxGapMetric(lb, ub) <= ApproxGapEps + 1e-9);
}

static bool GetStoredTauCertPair(
    const std::string& tauKey,
    double ubVal,
    double& ubOut,
    double& lbOut
) {
    ubOut = IloInfinity;
    lbOut = -IloInfinity;
    if (!std::isfinite(ubVal)) {
        return false;
    }
    auto itLb = TauFinalLBCache.find(tauKey);
    if (itLb == TauFinalLBCache.end() || !std::isfinite(itLb->second)) {
        return false;
    }
    ubOut = ubVal;
    lbOut = itLb->second;
    return true;
}

static void AccumulateMaskCertAggFromCache(
    int zMask,
    const std::unordered_map<std::string, double>& cacheTau,
    CertAggStats* certAgg
) {
    if (!certAgg || !SetupCertActive()) {
        return;
    }
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        const std::string& key = TauKeys[zMask][idx];
        auto itTau = cacheTau.find(key);
        if (itTau == cacheTau.end()) {
            continue;
        }
        double certUb = IloInfinity;
        double certLb = -IloInfinity;
        if (GetStoredTauCertPair(key, itTau->second, certUb, certLb)) {
            certAgg->finiteCount++;
            certAgg->ubSum += certUb;
            certAgg->lbSum += certLb;
        }
    }
}

static bool SameItemPattern(
    const std::vector<int>& yItem,
    const std::vector<std::vector<int>>& yPlan,
    int item
) {
    if (item < 0 || item >= N_it || (int)yItem.size() < N_tp + 1 || !IsValidYPlan(yPlan)) {
        return false;
    }
    for (int t = 1; t <= N_tp; ++t) {
        if (yItem[t] != yPlan[item][t]) {
            return false;
        }
    }
    return true;
}

static void RegisterTauReuseY(
    uint64_t tauMask,
    const std::vector<std::vector<int>>& yPlan,
    double obj,
    bool optimal,
    double certLB = -IloInfinity,
    bool certValid = false,
    bool certExact = false
) {
    if (tauMask == 0ULL || !std::isfinite(obj) || !IsValidYPlan(yPlan)) {
        return;
    }
    if (optimal && std::isfinite(obj)) {
        certLB = obj;
        certValid = true;
        certExact = true;
    }
    auto it = TauReuseYCache.find(tauMask);
    if (it == TauReuseYCache.end()) {
        TauReuseCandidate entry;
        it = TauReuseYCache.emplace(tauMask, entry).first;
        TauReuseYMasks.push_back(tauMask);
    }
    TauReuseCandidate& entry = it->second;
    if (!entry.hasY || obj < entry.obj - 1e-9) {
        entry.y = yPlan;
        entry.obj = obj;
        entry.optimal = optimal;
        entry.hasY = true;
    } else if (optimal && !entry.optimal) {
        entry.optimal = true;
    }
    if (certValid && std::isfinite(certLB)) {
        if (!entry.hasCertLB || certLB > entry.certLB + 1e-9) {
            entry.certLB = certLB;
            entry.hasCertLB = true;
            entry.certExact = certExact;
        } else if (std::fabs(certLB - entry.certLB) <= 1e-9 && certExact && !entry.certExact) {
            entry.certExact = true;
        }
    }
}

static bool BetterTauReuseRank(
    const TauReuseRankedCandidate& a,
    const TauReuseRankedCandidate& b
) {
    if (a.coverage > b.coverage + 1e-12) return true;
    if (a.coverage < b.coverage - 1e-12) return false;
    if (a.normObj < b.normObj - 1e-9) return true;
    if (a.normObj > b.normObj + 1e-9) return false;
    if (a.card > b.card) return true;
    if (a.card < b.card) return false;
    if (a.optimal && !b.optimal) return true;
    if (!a.optimal && b.optimal) return false;
    return false;
}

static bool CollectTopTauReuseY64(
    uint64_t targetMask,
    int topK,
    std::vector<TauReuseRankedCandidate>& out
) {
    out.clear();
    if (targetMask == 0ULL || TauReuseYMasks.empty()) {
        return false;
    }
    topK = std::max(1, topK);
    const int targetCard = std::max(1, __builtin_popcountll(targetMask));

    for (uint64_t candMask : TauReuseYMasks) {
        if (candMask == 0ULL || (candMask & ~targetMask) != 0ULL) {
            continue;
        }
        auto it = TauReuseYCache.find(candMask);
        if (it == TauReuseYCache.end()) {
            continue;
        }
        const TauReuseCandidate& cand = it->second;
        if (!cand.hasY || !std::isfinite(cand.obj) || !IsValidYPlan(cand.y)) {
            continue;
        }
        TauReuseRankedCandidate rc;
        rc.y = cand.y;
        rc.mask64 = candMask;
        rc.card = std::max(1, __builtin_popcountll(candMask));
        rc.coverage = static_cast<double>(rc.card) / static_cast<double>(targetCard);
        rc.normObj = cand.obj / static_cast<double>(rc.card);
        rc.optimal = cand.optimal;
        out.push_back(std::move(rc));
    }
    if (out.empty()) {
        return false;
    }
    std::sort(out.begin(), out.end(), BetterTauReuseRank);
    if ((int)out.size() > topK) {
        out.resize(topK);
    }
    return true;
}

static bool SelectBestTauReuseY(
    uint64_t targetMask,
    std::vector<std::vector<int>>& yBest,
    uint64_t* bestMaskOut = nullptr,
    int* bestCardOut = nullptr,
    double* bestCoverageOut = nullptr
) {
    yBest.clear();
    if (targetMask == 0ULL || TauReuseYMasks.empty()) {
        return false;
    }
    const int targetCard = std::max(1, __builtin_popcountll(targetMask));
    bool found = false;
    double bestCoverage = -1.0;
    double bestNormObj = IloInfinity;
    int bestCard = -1;
    uint64_t bestMask = 0ULL;
    bool bestOptimal = false;

    for (uint64_t candMask : TauReuseYMasks) {
        if (candMask == 0ULL || (candMask & ~targetMask) != 0ULL) {
            continue;
        }
        auto it = TauReuseYCache.find(candMask);
        if (it == TauReuseYCache.end()) {
            continue;
        }
        const TauReuseCandidate& cand = it->second;
        if (!cand.hasY || !std::isfinite(cand.obj) || !IsValidYPlan(cand.y)) {
            continue;
        }
        const int candCard = std::max(1, __builtin_popcountll(candMask));
        const double coverage = static_cast<double>(candCard) / static_cast<double>(targetCard);
        const double normObj = cand.obj / static_cast<double>(candCard);
        bool better = false;
        if (!found) {
            better = true;
        } else if (coverage > bestCoverage + 1e-12) {
            better = true;
        } else if (std::fabs(coverage - bestCoverage) <= 1e-12) {
            if (normObj < bestNormObj - 1e-9) {
                better = true;
            } else if (std::fabs(normObj - bestNormObj) <= 1e-9) {
                if (candCard > bestCard) {
                    better = true;
                } else if (candCard == bestCard && cand.optimal && !bestOptimal) {
                    better = true;
                }
            }
        }
        if (better) {
            found = true;
            bestCoverage = coverage;
            bestNormObj = normObj;
            bestCard = candCard;
            bestMask = candMask;
            bestOptimal = cand.optimal;
            yBest = cand.y;
        }
    }
    if (found) {
        if (bestMaskOut) *bestMaskOut = bestMask;
        if (bestCardOut) *bestCardOut = bestCard;
        if (bestCoverageOut) *bestCoverageOut = bestCoverage;
    }
    return found;
}

static void PbListToMaskBlocks(
    const std::vector<int>& pbList,
    std::vector<uint64_t>& bitsOut,
    int& cardOut
) {
    const int blocks = std::max(1, (N_pb + 63) / 64);
    bitsOut.assign(blocks, 0ULL);
    cardOut = 0;
    for (int pb : pbList) {
        if (pb < 0 || pb >= N_pb) {
            continue;
        }
        bitsOut[pb >> 6] |= (1ULL << static_cast<unsigned int>(pb & 63));
        cardOut++;
    }
}

static bool BitsSubsetDyn(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b
) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if ((a[i] & ~b[i]) != 0ULL) {
            return false;
        }
    }
    for (size_t i = n; i < a.size(); ++i) {
        if (a[i] != 0ULL) {
            return false;
        }
    }
    return true;
}

static bool BitsDisjointDyn(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b
) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if ((a[i] & b[i]) != 0ULL) {
            return false;
        }
    }
    return true;
}

static void BitsMinusDyn(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b,
    std::vector<uint64_t>& out
) {
    out = a;
    const size_t n = std::min(out.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        out[i] &= ~b[i];
    }
}

static void BitsToPbList(
    const std::vector<uint64_t>& bits,
    std::vector<int>& pbList
) {
    pbList.clear();
    for (size_t blk = 0; blk < bits.size(); ++blk) {
        uint64_t word = bits[blk];
        while (word) {
            uint64_t bit = word & (~word + 1ULL);
            int off = __builtin_ctzll(bit);
            int pb = static_cast<int>(blk * 64 + static_cast<size_t>(off));
            if (pb < N_pb) {
                pbList.push_back(pb);
            }
            word ^= bit;
        }
    }
}

static void RegisterTauReuseY_Dyn(
    const std::string& tauKey,
    const std::vector<int>& pbList,
    const std::vector<std::vector<int>>& yPlan,
    double obj,
    bool optimal,
    double certLB = -IloInfinity,
    bool certValid = false,
    bool certExact = false
) {
    if (!std::isfinite(obj) || !IsValidYPlan(yPlan)) {
        return;
    }
    if (optimal && std::isfinite(obj)) {
        certLB = obj;
        certValid = true;
        certExact = true;
    }
    auto itIdx = TauReuseDynIndexByKey.find(tauKey);
    size_t idx = 0;
    if (itIdx == TauReuseDynIndexByKey.end()) {
        TauReuseEntryDyn entry;
        PbListToMaskBlocks(pbList, entry.bits, entry.card);
        idx = TauReuseDynEntries.size();
        TauReuseDynEntries.push_back(entry);
        TauReuseDynIndexByKey.emplace(tauKey, idx);
    } else {
        idx = itIdx->second;
    }
    TauReuseEntryDyn& entry = TauReuseDynEntries[idx];
    if (!entry.hasY || obj < entry.obj - 1e-9) {
        entry.y = yPlan;
        entry.obj = obj;
        entry.optimal = optimal;
        entry.hasY = true;
    } else if (optimal && !entry.optimal) {
        entry.optimal = true;
    }
    if (certValid && std::isfinite(certLB)) {
        if (!entry.hasCertLB || certLB > entry.certLB + 1e-9) {
            entry.certLB = certLB;
            entry.hasCertLB = true;
            entry.certExact = certExact;
        } else if (std::fabs(certLB - entry.certLB) <= 1e-9 && certExact && !entry.certExact) {
            entry.certExact = true;
        }
    }
}

static bool SelectBestTauReuseY_Dyn(
    const std::vector<int>& pbListTarget,
    std::vector<std::vector<int>>& yBest,
    std::vector<uint64_t>* bestBitsOut = nullptr,
    int* bestCardOut = nullptr,
    double* bestCoverageOut = nullptr
) {
    yBest.clear();
    if (TauReuseDynEntries.empty()) {
        return false;
    }
    std::vector<uint64_t> targetBits;
    int targetCard = 0;
    PbListToMaskBlocks(pbListTarget, targetBits, targetCard);
    targetCard = std::max(1, targetCard);

    bool found = false;
    double bestCoverage = -1.0;
    double bestNormObj = IloInfinity;
    int bestCard = -1;
    std::vector<uint64_t> bestBits;
    bool bestOptimal = false;

    for (const TauReuseEntryDyn& cand : TauReuseDynEntries) {
        if (!cand.hasY || !std::isfinite(cand.obj) || !IsValidYPlan(cand.y)) {
            continue;
        }
        if (!BitsSubsetDyn(cand.bits, targetBits)) {
            continue;
        }
        const int candCard = std::max(1, cand.card);
        const double coverage = static_cast<double>(candCard) / static_cast<double>(targetCard);
        const double normObj = cand.obj / static_cast<double>(candCard);
        bool better = false;
        if (!found) {
            better = true;
        } else if (coverage > bestCoverage + 1e-12) {
            better = true;
        } else if (std::fabs(coverage - bestCoverage) <= 1e-12) {
            if (normObj < bestNormObj - 1e-9) {
                better = true;
            } else if (std::fabs(normObj - bestNormObj) <= 1e-9) {
                if (candCard > bestCard) {
                    better = true;
                } else if (candCard == bestCard && cand.optimal && !bestOptimal) {
                    better = true;
                }
            }
        }
        if (better) {
            found = true;
            bestCoverage = coverage;
            bestNormObj = normObj;
            bestCard = candCard;
            bestBits = cand.bits;
            bestOptimal = cand.optimal;
            yBest = cand.y;
        }
    }
    if (found) {
        if (bestBitsOut) *bestBitsOut = bestBits;
        if (bestCardOut) *bestCardOut = bestCard;
        if (bestCoverageOut) *bestCoverageOut = bestCoverage;
    }
    return found;
}

static bool CollectTopTauReuseYDyn(
    const std::vector<int>& pbListTarget,
    int topK,
    std::vector<TauReuseRankedCandidate>& out
) {
    out.clear();
    if (TauReuseDynEntries.empty()) {
        return false;
    }
    topK = std::max(1, topK);
    std::vector<uint64_t> targetBits;
    int targetCard = 0;
    PbListToMaskBlocks(pbListTarget, targetBits, targetCard);
    targetCard = std::max(1, targetCard);

    for (const TauReuseEntryDyn& cand : TauReuseDynEntries) {
        if (!cand.hasY || !std::isfinite(cand.obj) || !IsValidYPlan(cand.y)) {
            continue;
        }
        if (!BitsSubsetDyn(cand.bits, targetBits)) {
            continue;
        }
        TauReuseRankedCandidate rc;
        rc.y = cand.y;
        rc.bits = cand.bits;
        rc.card = std::max(1, cand.card);
        rc.coverage = static_cast<double>(rc.card) / static_cast<double>(targetCard);
        rc.normObj = cand.obj / static_cast<double>(rc.card);
        rc.optimal = cand.optimal;
        out.push_back(std::move(rc));
    }
    if (out.empty()) {
        return false;
    }
    std::sort(out.begin(), out.end(), BetterTauReuseRank);
    if ((int)out.size() > topK) {
        out.resize(topK);
    }
    return true;
}

static bool BuildConsensusFreeComponents(
    const std::vector<TauReuseRankedCandidate>& candidates,
    std::vector<std::vector<char>>& freeYOut
) {
    freeYOut.assign(N_it, std::vector<char>(N_tp + 1, 0));
    if (candidates.size() < 2 || N_tp <= 0 || N_it <= 0) {
        return false;
    }

    struct FreeComp {
        double score;
        int it;
        int t;
    };

    const int K = static_cast<int>(candidates.size());
    std::vector<FreeComp> rank;
    rank.reserve(N_it * N_tp);
    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int ones = 0;
            for (int k = 0; k < K; ++k) {
                if (candidates[k].y[it][t] == 1) {
                    ones++;
                }
            }
            int zeros = K - ones;
            if (ones == 0 || zeros == 0) {
                continue;
            }
            double disagree = static_cast<double>(std::min(ones, zeros)) / static_cast<double>(std::max(1, K));
            rank.push_back({disagree, it, t});
        }
    }

    if (rank.empty()) {
        return false;
    }

    std::sort(rank.begin(), rank.end(), [](const FreeComp& a, const FreeComp& b) {
        if (a.score == b.score) {
            if (a.t == b.t) return a.it < b.it;
            return a.t < b.t;
        }
        return a.score > b.score;
    });

    int maxFreeVars = std::max(1, static_cast<int>(std::ceil(0.10 * static_cast<double>(N_it * N_tp))));
    maxFreeVars = std::min(maxFreeVars, 8 * std::max(1, N_it));
    maxFreeVars = std::min(maxFreeVars, static_cast<int>(rank.size()));
    for (int k = 0; k < maxFreeVars; ++k) {
        freeYOut[rank[k].it][rank[k].t] = 1;
    }
    return maxFreeVars > 0;
}

static bool BuildDeltaFreePeriodsFromReuse(
    const std::vector<int>& pbListTarget,
    const std::vector<std::vector<int>>& yAnchor,
    uint64_t sourceMask64,
    const std::vector<uint64_t>* sourceBitsDyn,
    std::vector<char>& freeTOut,
    double* coverageOut = nullptr
) {
    freeTOut.assign(N_tp + 1, 0);
    if (pbListTarget.empty() || N_tp <= 0) {
        if (coverageOut) *coverageOut = 1.0;
        return false;
    }

    auto inSource = [&](int pb) -> bool {
        if (pb < 0 || pb >= N_pb) return false;
        if (sourceBitsDyn && !sourceBitsDyn->empty()) {
            int blk = pb >> 6;
            if (blk >= (int)sourceBitsDyn->size()) return false;
            return (((*sourceBitsDyn)[blk] >> (pb & 63)) & 1ULL) != 0ULL;
        }
        if (pb >= 64) return false;
        return ((sourceMask64 >> pb) & 1ULL) != 0ULL;
    };

    std::vector<int> pbDelta;
    pbDelta.reserve(pbListTarget.size());
    int sourceCount = 0;
    for (int pb : pbListTarget) {
        if (inSource(pb)) sourceCount++;
        else pbDelta.push_back(pb);
    }

    const int targetCount = std::max(1, (int)pbListTarget.size());
    const double coverage = static_cast<double>(sourceCount) / static_cast<double>(targetCount);
    if (coverageOut) *coverageOut = coverage;
    if (pbDelta.empty()) {
        return false;
    }

    // Only intensify when reused candidate covers a limited part of the target subproblem.
    const double coverageThreshold = 0.80;
    if (coverage >= coverageThreshold) {
        return false;
    }

    const double capD = std::max(1.0, static_cast<double>(Cap));
    std::vector<std::pair<double, int>> rank;
    rank.reserve(N_tp);
    for (int t = 1; t <= N_tp; ++t) {
        double demDelta = 0.0;
        for (int pb : pbDelta) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int it = 0; it < N_it; ++it) {
                    demDelta += d[t][sc][pb][it];
                }
            }
        }
        demDelta /= std::max(1, (int)pbDelta.size() * N_sc);

        int yOn = 0;
        for (int it = 0; it < N_it; ++it) {
            if (it < (int)yAnchor.size() && t < (int)yAnchor[it].size() && yAnchor[it][t] == 1) {
                yOn++;
            }
        }
        double yOnRatio = static_cast<double>(yOn) / std::max(1, N_it);
        double score = (demDelta / capD) + 0.35 * yOnRatio;
        rank.push_back(std::make_pair(score, t));
    }

    std::sort(rank.begin(), rank.end(), [](const std::pair<double, int>& a,
                                           const std::pair<double, int>& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first > b.first;
    });

    double freeFrac = (coverage < 0.5) ? 0.20 : 0.10;
    int maxFreePeriods = std::max(1, static_cast<int>(std::ceil(freeFrac * static_cast<double>(N_tp))));
    maxFreePeriods = std::min(maxFreePeriods, 8);

    for (int k = 0; k < maxFreePeriods && k < (int)rank.size(); ++k) {
        freeTOut[rank[k].second] = 1;
    }
    return true;
}

static double SolveTauJointSubproblem_WarmStart(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    const std::vector<std::vector<int>>* yStart,
    bool* optimalOut = nullptr,
    std::vector<std::vector<int>>* yBestOut = nullptr,
    double externalLb = -IloInfinity,
    bool* targetHitOut = nullptr
) {
    const int pbCount = static_cast<int>(pbList.size());
    if (targetHitOut) {
        *targetHitOut = false;
    }
    if (yBestOut) {
        yBestOut->assign(N_it, std::vector<int>(N_tp + 1, 0));
    }
    if (pbCount == 0) {
        if (optimalOut) {
            *optimalOut = true;
        }
        return 0.0;
    }
    IloNum rho_joint = static_cast<IloNum>(1) / (static_cast<IloNum>(N_pb) * static_cast<IloNum>(N_sc));

    IloNumVarArray2 X(env, N_it);
    IloNumVarArray2 Y(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        X[it] = IloNumVarArray(env, N_tp + 1, 0, IloInfinity, ILOFLOAT);
        Y[it] = IloNumVarArray(env, N_tp + 1, 0, 1, ILOBOOL);
    }

    IloNumVarArray3 I(env, N_it);
    IloNumVarArray3 L(env, N_it);
    for (int it = 0; it < N_it; ++it) {
        I[it] = IloNumVarArray2(env, N_sc);
        L[it] = IloNumVarArray2(env, N_sc);
        for (int sc = 0; sc < N_sc; ++sc) {
            I[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
            L[it][sc] = IloNumVarArray(env, pbCount * (N_tp + 1), 0, IloInfinity, ILOFLOAT);
        }
    }

    IloModel model(env);
    IloCplex cplex(model);

    double pbWeight = static_cast<double>(pbCount) / static_cast<double>(N_pb);
    IloExpr obj(env);
    for (int t = 1; t <= N_tp; ++t) {
        for (int it = 0; it < N_it; ++it) {
            obj += pbWeight * (p[t][it] * X[it][t] + f[t][it] * Y[it][t]);
        }
    }
    for (int k = 0; k < pbCount; ++k) {
        int pb = pbList[k];
        for (int it = 0; it < N_it; ++it) {
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    obj += rho_joint * (h[t][it] * I[it][sc][idx] + b[t][it] * L[it][sc][idx]);
                }
            }
        }
    }
    model.add(IloMinimize(env, obj));

    for (int it = 0; it < N_it; ++it) {
        for (int t = 1; t <= N_tp; ++t) {
            int minBigM = Cap;
            for (int pb : pbList) {
                minBigM = std::min(minBigM, (int)BigM[it][pb][t]);
            }
            model.add(X[it][t] <= std::min(Cap, minBigM) * Y[it][t]);
        }
        model.add(X[it][0] == 0);
        model.add(Y[it][0] == 0);
    }

    for (int t = 1; t <= N_tp; ++t) {
        IloExpr capExpr(env);
        for (int it = 0; it < N_it; ++it) {
            capExpr += X[it][t];
        }
        model.add(capExpr <= Cap);
        capExpr.end();
    }

    for (int it = 0; it < N_it; ++it) {
        for (int k = 0; k < pbCount; ++k) {
            int pb = pbList[k];
            for (int sc = 0; sc < N_sc; ++sc) {
                for (int t = 1; t <= N_tp; ++t) {
                    int idx = k * (N_tp + 1) + t;
                    int idxPrev = k * (N_tp + 1) + (t - 1);
                    model.add(I[it][sc][idx] == I[it][sc][idxPrev] + X[it][t] - d[t][sc][pb][it] + L[it][sc][idx]);
                }
                int idx0 = k * (N_tp + 1);
                model.add(I[it][sc][idx0] == 0);
                model.add(L[it][sc][idx0] == 0);
            }
        }
    }

    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
    cplex.setParam(IloCplex::Threads, 2);
    cplex.setParam(IloCplex::TiLim, 900);
    if (std::isfinite(externalLb)) {
        cplex.setParam(IloCplex::MIPEmphasis, IloCplex::MIPEmphasisFeasibility);
        cplex.use(new (env) StopOnApproxGapIncumbentI(env, externalLb, std::max(0.0, ApproxGapEps)));
    }
    if (VI_pro1 == 1 || VI_pro2 == 1) {
        cplex.use(new (env) TauJointValidIneqCallback(Y, I, L, pbList, zMask),
                  IloCplex::Callback::Context::Id::Relaxation);
    }

    if (yStart && IsValidYPlan(*yStart)) {
        IloNumVarArray startVars(env);
        IloNumArray startVals(env);
        for (int it = 0; it < N_it; ++it) {
            for (int t = 1; t <= N_tp; ++t) {
                startVars.add(Y[it][t]);
                startVals.add(((*yStart)[it][t] >= 1) ? 1.0 : 0.0);
            }
        }
        if (startVars.getSize() > 0) {
            cplex.addMIPStart(startVars, startVals, IloCplex::MIPStartAuto);
        }
    }

    if (!cplex.solve()) {
        if (optimalOut) {
            *optimalOut = false;
        }
        obj.end();
        cplex.end();
        model.end();
        return IloInfinity;
    }

    if (optimalOut) {
        IloAlgorithm::Status status = cplex.getStatus();
        *optimalOut = (status == IloAlgorithm::Optimal);
    }
    double objVal = cplex.getObjValue();
    if (targetHitOut && std::isfinite(externalLb)) {
        double lbDyn = externalLb;
        double bestBound = cplex.getBestObjValue();
        if (std::isfinite(bestBound) && (!std::isfinite(lbDyn) || bestBound > lbDyn)) {
            lbDyn = bestBound;
        }
        *targetHitOut = ApproxGapSatisfiedWithLb(lbDyn, objVal, std::max(0.0, ApproxGapEps));
    }
    if (yBestOut) {
        for (int it = 0; it < N_it; ++it) {
            for (int t = 0; t <= N_tp; ++t) {
                (*yBestOut)[it][t] = (cplex.getValue(Y[it][t]) >= 0.5) ? 1 : 0;
            }
        }
    }

    obj.end();
    cplex.end();
    model.end();
    return objVal;
}

static TauLagResult SolveTauRLWithSeed(
    IloEnv env,
    const std::vector<int>& pbList,
    int zMask,
    int maxIter,
    double step0
){
    const int pbCount = static_cast<int>(pbList.size());
    if (pbCount == 0) {
        TauLagResult out;
        out.lb = 0.0;
        out.ub = 0.0;
        out.finalSeed.valid = true;
        out.finalSeed.y.assign(N_it, std::vector<int>(N_tp + 1, 0));
        out.finalSeed.x.assign(N_it, std::vector<double>(N_tp + 1, 0.0));
        out.finalSeed.itemObjLB.assign(N_it, IloInfinity);
        out.finalSeed.sumX.assign(N_tp + 1, 0.0);
        out.finalSeed.actualObj = 0.0;
        out.finalSeed.lagObj = 0.0;
        return out;
    }
    std::vector<double> lambda(N_tp + 1, 0.0);
    std::unordered_set<std::string> seenY;
    TauLagResult out;

    for (int iter = 0; iter < maxIter; ++iter) {
        TauHeuristicSeed seedIter;
        if (!BuildTauHeuristicSeedFromLambda(env, pbList, zMask, lambda, nullptr, false, seedIter)) {
            break;
        }
        out.finalSeed = seedIter;

        double dual = seedIter.lagObj;
        for (int t = 1; t <= N_tp; ++t) {
            dual -= lambda[t] * Cap;
        }
        if (dual > out.lb) {
            out.lb = dual;
        }

        std::string key;
        key.reserve(N_it * (N_tp + 1) * 2);
        for (int i = 0; i < N_it; ++i) {
            for (int t = 1; t <= N_tp; ++t) {
                key.push_back(seedIter.y[i][t] ? '1' : '0');
            }
            key.push_back('|');
        }
        if (seenY.insert(key).second) {
            double ub = SolveTauJointFixedY_LP(env, pbList, seedIter.y);
            if (ub < out.ub) {
                out.ub = ub;
                out.bestUbY = seedIter.y;
                out.hasBestUbY = true;
            }
        }

        bool feasible = true;
        double norm = 0.0;
        for (int t = 1; t <= N_tp; ++t) {
            double g = seedIter.sumX[t] - Cap;
            if (g > 1e-6) feasible = false;
            if (g > 0.0) norm += g * g;
        }
        if (feasible) break;
        if (norm > 0.0) {
            double step = step0 / std::sqrt(static_cast<double>(iter + 1));
            for (int t = 1; t <= N_tp; ++t) {
                double g = seedIter.sumX[t] - Cap;
                if (g > 0.0) {
                    lambda[t] = std::max(0.0, lambda[t] + step * g);
                }
            }
        }
    }
    if (!std::isfinite(out.ub) && out.finalSeed.valid && IsValidYPlan(out.finalSeed.y)) {
        out.ub = SolveTauJointFixedY_LP(env, pbList, out.finalSeed.y);
        out.bestUbY = out.finalSeed.y;
        out.hasBestUbY = true;
    }
    return out;
}

static RLObjective SolveTauRL(
    IloEnv env,
    const std::vector<int>& pbList,
    int maxIter,
    double step0
){
    TauLagResult out = SolveTauRLWithSeed(env, pbList, -1, maxIter, step0);
    return {out.lb, out.ub};
}

static double ComputeTauCompetingSetupLagLB(
    IloEnv env,
    const std::vector<int>& pbList,
    const std::vector<std::vector<int>>& yBar,
    const std::vector<double>& lambdaInit,
    EvalStats* stats
) {
    if (!IsValidYPlan(yBar) || pbList.empty()) {
        return -IloInfinity;
    }

    std::vector<double> lambda = lambdaInit;
    if ((int)lambda.size() < N_tp + 1) {
        lambda.assign(N_tp + 1, 0.0);
    }

    double bestLB = -IloInfinity;
    const int maxIter = std::max(1, SetupCertLagMaxIter);
    const double step0 = std::max(1e-6, SetupCertLagStep0);

    for (int iter = 0; iter < maxIter; ++iter) {
        if (BBTimeLimitReached()) {
            break;
        }
        std::vector<double> sumX(N_tp + 1, 0.0);
        double sumStar = 0.0;
        double bestDelta = IloInfinity;
        bool ok = true;

        for (int it = 0; it < N_it; ++it) {
            std::vector<int> yStar;
            std::vector<double> xStar;
            auto t0 = std::chrono::steady_clock::now();
            double vStar = SolveTauItemSubproblem_LR(
                env, it, pbList, lambda, yStar, xStar
            );
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->itemSolved += 1;
                stats->timeItem += dt;
            }
            if (!std::isfinite(vStar)) {
                ok = false;
                break;
            }
            sumStar += vStar;
            for (int t = 1; t <= N_tp; ++t) {
                sumX[t] += xStar[t];
            }

            double vNeq = vStar;
            if (SameItemPattern(yStar, yBar, it)) {
                std::vector<int> yAlt;
                std::vector<double> xAlt;
                t0 = std::chrono::steady_clock::now();
                vNeq = SolveTauItemSubproblem_LR_ExcludePattern(
                    env, it, pbList, lambda, yBar[it], yAlt, xAlt, SetupCertItemTiLim
                );
                dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->itemSolved += 1;
                    stats->timeItem += dt;
                }
            }
            if (std::isfinite(vNeq)) {
                bestDelta = std::min(bestDelta, vNeq - vStar);
            }
        }

        if (!ok || !std::isfinite(bestDelta)) {
            break;
        }

        double dualComp = sumStar + bestDelta;
        for (int t = 1; t <= N_tp; ++t) {
            dualComp -= lambda[t] * Cap;
        }
        if (dualComp > bestLB) {
            bestLB = dualComp;
        }

        double norm = 0.0;
        for (int t = 1; t <= N_tp; ++t) {
            double g = sumX[t] - Cap;
            if (g > 0.0) {
                norm += g * g;
            }
        }
        if (norm <= 1e-12) {
            break;
        }
        double step = step0 / std::sqrt(static_cast<double>(iter + 1));
        for (int t = 1; t <= N_tp; ++t) {
            double g = sumX[t] - Cap;
            if (g > 0.0) {
                lambda[t] = std::max(0.0, lambda[t] + step * g);
            }
        }
    }

    return bestLB;
}

struct TauCertPackCandidate64 {
    uint64_t mask = 0ULL;
    int card = 0;
    double lb = -IloInfinity;
    bool exact = false;
};

struct TauCertPackCandidateDyn {
    std::vector<uint64_t> bits;
    int card = 0;
    double lb = -IloInfinity;
    bool exact = false;
};

static double ComputeCertifiedPackingLb64(
    uint64_t targetMask,
    std::vector<int>& residualPbList
) {
    residualPbList.clear();
    if (targetMask == 0ULL) {
        return 0.0;
    }

    std::vector<TauCertPackCandidate64> cands;
    for (uint64_t candMask : TauReuseYMasks) {
        if (candMask == 0ULL || (candMask & ~targetMask) != 0ULL) {
            continue;
        }
        auto it = TauReuseYCache.find(candMask);
        if (it == TauReuseYCache.end()) {
            continue;
        }
        const TauReuseCandidate& cand = it->second;
        if (!cand.hasCertLB || !std::isfinite(cand.certLB)) {
            continue;
        }
        TauCertPackCandidate64 entry;
        entry.mask = candMask;
        entry.card = std::max(1, __builtin_popcountll(candMask));
        entry.lb = cand.certLB;
        entry.exact = cand.certExact;
        cands.push_back(entry);
    }
    if (cands.empty()) {
        Mask64ToPbList(targetMask, residualPbList);
        return -IloInfinity;
    }

    uint64_t coveredMask = 0ULL;
    double best = 0.0;
    const int targetCard = __builtin_popcountll(targetMask);
    const int kExactPackMaxCard = 18;
    if (targetCard <= kExactPackMaxCard) {
        std::vector<int> targetBits;
        targetBits.reserve(targetCard);
        uint64_t tmp = targetMask;
        while (tmp) {
            uint64_t bit = tmp & (~tmp + 1ULL);
            targetBits.push_back(__builtin_ctzll(bit));
            tmp ^= bit;
        }
        struct LocalCand {
            uint64_t localMask = 0ULL;
            uint64_t globalMask = 0ULL;
            double lb = -IloInfinity;
            bool exact = false;
        };
        std::unordered_map<uint64_t, LocalCand> bestLocal;
        for (const TauCertPackCandidate64& cand : cands) {
            uint64_t localMask = 0ULL;
            for (int pos = 0; pos < targetCard; ++pos) {
                if ((cand.mask >> static_cast<unsigned int>(targetBits[pos])) & 1ULL) {
                    localMask |= (1ULL << static_cast<unsigned int>(pos));
                }
            }
            auto it = bestLocal.find(localMask);
            if (it == bestLocal.end() || cand.lb > it->second.lb + 1e-9 ||
                (std::fabs(cand.lb - it->second.lb) <= 1e-9 && cand.exact && !it->second.exact)) {
                LocalCand lc;
                lc.localMask = localMask;
                lc.globalMask = cand.mask;
                lc.lb = cand.lb;
                lc.exact = cand.exact;
                bestLocal[localMask] = lc;
            }
        }
        const uint64_t stateCount = 1ULL << static_cast<unsigned int>(targetCard);
        std::vector<double> dp(stateCount, -IloInfinity);
        std::vector<long long> prev(stateCount, -1);
        std::vector<uint64_t> chosen(stateCount, 0ULL);
        dp[0] = 0.0;
        for (const auto& kv : bestLocal) {
            const LocalCand& cand = kv.second;
            for (long long s = static_cast<long long>(stateCount) - 1; s >= 0; --s) {
                if (!std::isfinite(dp[static_cast<size_t>(s)])) {
                    continue;
                }
                if ((static_cast<uint64_t>(s) & cand.localMask) != 0ULL) {
                    continue;
                }
                uint64_t next = static_cast<uint64_t>(s) | cand.localMask;
                double candVal = dp[static_cast<size_t>(s)] + cand.lb;
                if (candVal > dp[static_cast<size_t>(next)] + 1e-9) {
                    dp[static_cast<size_t>(next)] = candVal;
                    prev[static_cast<size_t>(next)] = s;
                    chosen[static_cast<size_t>(next)] = cand.globalMask;
                }
            }
        }
        uint64_t bestState = 0ULL;
        for (uint64_t s = 0; s < stateCount; ++s) {
            if (std::isfinite(dp[static_cast<size_t>(s)]) && dp[static_cast<size_t>(s)] > best + 1e-9) {
                best = dp[static_cast<size_t>(s)];
                bestState = s;
            }
        }
        while (bestState != 0ULL && prev[static_cast<size_t>(bestState)] >= 0) {
            coveredMask |= chosen[static_cast<size_t>(bestState)];
            bestState = static_cast<uint64_t>(prev[static_cast<size_t>(bestState)]);
        }
    } else {
        std::sort(cands.begin(), cands.end(),
                  [](const TauCertPackCandidate64& a, const TauCertPackCandidate64& b) {
                      if (a.card != b.card) return a.card > b.card;
                      if (a.lb != b.lb) return a.lb > b.lb;
                      if (a.exact != b.exact) return a.exact;
                      return a.mask < b.mask;
                  });
        for (const TauCertPackCandidate64& cand : cands) {
            if ((cand.mask & coveredMask) != 0ULL) {
                continue;
            }
            coveredMask |= cand.mask;
            best += cand.lb;
        }
    }

    Mask64ToPbList(targetMask & ~coveredMask, residualPbList);
    return best;
}

static double ComputeCertifiedPackingLbDyn(
    const std::vector<int>& pbListTarget,
    std::vector<int>& residualPbList
) {
    residualPbList.clear();
    if (pbListTarget.empty()) {
        return 0.0;
    }

    std::vector<uint64_t> targetBits;
    int targetCard = 0;
    PbListToMaskBlocks(pbListTarget, targetBits, targetCard);
    std::vector<TauCertPackCandidateDyn> cands;
    for (const TauReuseEntryDyn& cand : TauReuseDynEntries) {
        if (!cand.hasCertLB || !std::isfinite(cand.certLB)) {
            continue;
        }
        if (!BitsSubsetDyn(cand.bits, targetBits)) {
            continue;
        }
        TauCertPackCandidateDyn entry;
        entry.bits = cand.bits;
        entry.card = std::max(1, cand.card);
        entry.lb = cand.certLB;
        entry.exact = cand.certExact;
        cands.push_back(std::move(entry));
    }
    if (cands.empty()) {
        residualPbList = pbListTarget;
        return -IloInfinity;
    }

    std::sort(cands.begin(), cands.end(),
              [](const TauCertPackCandidateDyn& a, const TauCertPackCandidateDyn& b) {
                  if (a.card != b.card) return a.card > b.card;
                  if (a.lb != b.lb) return a.lb > b.lb;
                  if (a.exact != b.exact) return a.exact;
                  return a.bits < b.bits;
              });
    std::vector<uint64_t> covered(targetBits.size(), 0ULL);
    double best = 0.0;
    for (const TauCertPackCandidateDyn& cand : cands) {
        if (!BitsDisjointDyn(cand.bits, covered)) {
            continue;
        }
        for (size_t i = 0; i < covered.size() && i < cand.bits.size(); ++i) {
            covered[i] |= cand.bits[i];
        }
        best += cand.lb;
    }

    std::vector<uint64_t> residualBits;
    BitsMinusDyn(targetBits, covered, residualBits);
    BitsToPbList(residualBits, residualPbList);
    return best;
}

static double ComputeCertifiedPackingLb(
    const std::vector<int>& pbList,
    int /*zMask*/,
    EvalStats* /*stats*/
) {
    if (pbList.empty()) {
        return 0.0;
    }

    std::vector<int> residualPbList;
    double packLB = -IloInfinity;
    if (N_pb <= 63) {
        uint64_t mask = 0ULL;
        if (!PbListToMask64(pbList, mask)) {
            return -IloInfinity;
        }
        packLB = ComputeCertifiedPackingLb64(mask, residualPbList);
    } else {
        packLB = ComputeCertifiedPackingLbDyn(pbList, residualPbList);
    }
    if (!std::isfinite(packLB)) {
        return -IloInfinity;
    }
    (void)residualPbList;
    return packLB;
}

static TauRelaxedLPInfo GetTauRelaxedLPInfoCached(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    EvalStats* stats
) {
    auto itLP = TauRelaxedLPCache.find(tauKey);
    if (itLP != TauRelaxedLPCache.end()) {
        return itLP->second;
    }
    auto t0 = std::chrono::steady_clock::now();
    TauRelaxedLPInfo lpInfo = SolveTauJointRelaxedY_LPWithDuals(env, pbList);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stats) {
        stats->jointSolved += 1;
        stats->timeJoint += dt;
    }
    TauRelaxedLPCache[tauKey] = lpInfo;
    return lpInfo;
}

static double ComputeTauApproxGlobalLagLb(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    EvalStats* stats
) {
    auto it = TauApproxGlobalLBCache.find(tauKey);
    if (it != TauApproxGlobalLBCache.end()) {
        return it->second;
    }
    auto t0 = std::chrono::steady_clock::now();
    RLObjective rl = SolveTauRL(
        env,
        pbList,
        std::max(1, ApproxGlobalLagMaxIter),
        std::max(1e-6, ApproxGlobalLagStep0)
    );
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (stats) {
        stats->jointSolved += 1;
        stats->timeJoint += dt;
    }
    TauApproxGlobalLBCache[tauKey] = rl.lb;
    return rl.lb;
}

static double ComputeTauApproxLb(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    EvalStats* stats,
    bool includeGlobalLag = true
) {
    double lb = -IloInfinity;
    TauRelaxedLPInfo lpInfo = GetTauRelaxedLPInfoCached(env, tauKey, pbList, stats);
    if (lpInfo.valid && std::isfinite(lpInfo.lpLB)) {
        lb = lpInfo.lpLB;
    }
    if (includeGlobalLag && ApproxGlobalLagLBMode != 0) {
        double lagLb = ComputeTauApproxGlobalLagLb(env, tauKey, pbList, stats);
        if (std::isfinite(lagLb) && lagLb > lb) {
            lb = lagLb;
        }
    }
    return lb;
}

static double ComputeTauGlobalLb(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    int zMask,
    EvalStats* stats,
    TauRelaxedLPInfo* lpInfoOut = nullptr
) {
    TauRelaxedLPInfo lpInfo = GetTauRelaxedLPInfoCached(env, tauKey, pbList, stats);
    if (lpInfoOut) {
        *lpInfoOut = lpInfo;
    }

    double lb = -IloInfinity;
    if (lpInfo.valid && std::isfinite(lpInfo.lpLB)) {
        lb = lpInfo.lpLB;
    }
    double packLB = ComputeCertifiedPackingLb(pbList, zMask, stats);
    if (std::isfinite(packLB) && packLB > lb) {
        lb = packLB;
    }
    return lb;
}

static bool TryCertifyTauSetupCandidate(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    int zMask,
    const std::vector<std::vector<int>>& yBar,
    double ubCand,
    EvalStats* stats,
    bool& exactCertified,
    double& lbOut
) {
    exactCertified = false;
    lbOut = -IloInfinity;
    if (!SetupCertActive() || !std::isfinite(ubCand) || !IsValidYPlan(yBar)) {
        return false;
    }

    SetupCertCheckCount++;

    TauRelaxedLPInfo lpInfo = GetTauRelaxedLPInfoCached(env, tauKey, pbList, stats);

    std::string compKey = tauKey + "#cert#" + YPlanSignature(yBar);
    auto itComp = TauCertCompLagCache.find(compKey);
    double lagLB = -IloInfinity;
    if (itComp != TauCertCompLagCache.end()) {
        lagLB = itComp->second;
    } else if (lpInfo.valid) {
        lagLB = ComputeTauCompetingSetupLagLB(env, pbList, yBar, lpInfo.lambda, stats);
        TauCertCompLagCache[compKey] = lagLB;
    }
    double lb = lagLB;
    if (std::isfinite(lb)) {
        SetupCertMaxLagCount++;
        SetupCertFiniteLBCount++;
        SetupCertFiniteUBSum += ubCand;
        SetupCertFiniteLBSum += lb;
    }
    lbOut = lb;
    if (!std::isfinite(lb)) {
        return false;
    }

    auto countWithinTau = [&](double tau, int idx) {
        const double rhsTau = (1.0 + std::max(0.0, tau)) * lb;
        const double epsTau = 1e-6 * std::max(1.0, std::fabs(rhsTau));
        if (ubCand <= rhsTau + epsTau) {
            SetupCertCoverageCounts[idx]++;
            SetupCertCoverageUbSums[idx] += ubCand;
            SetupCertCoverageLbSums[idx] += lb;
        }
    };
    for (int i = 0; i <= 10; ++i) {
        countWithinTau(0.01 * static_cast<double>(i), i);
    }

    const double rhs = IsApproxCertWht() ? ((1.0 + std::max(0.0, SetupCertTau)) * lb) : lb;
    const double eps = 1e-6 * std::max(1.0, std::fabs(rhs));
    if (ubCand <= rhs + eps) {
        exactCertified = !IsApproxCertWht();
        return true;
    }
    return false;
}

static void FinalizeTauCandidateWithCertification(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    int zMask,
    std::vector<std::vector<int>>& yCand,
    double& val,
    EvalStats* stats,
    bool& certAccepted,
    bool& exactAccepted,
    double& certLB,
    bool& hasFinalLB,
    double& finalLB
) {
    certAccepted = false;
    exactAccepted = false;
    certLB = -IloInfinity;
    hasFinalLB = false;
    finalLB = -IloInfinity;
    if (!SetupCertActive() || !std::isfinite(val) || !IsValidYPlan(yCand)) {
        return;
    }

    certAccepted = TryCertifyTauSetupCandidate(
        env, tauKey, pbList, zMask, yCand, val, stats, exactAccepted, certLB
    );
    if (std::isfinite(certLB)) {
        hasFinalLB = true;
        finalLB = certLB;
    }

    if (!certAccepted && IsApproxCertWht()) {
        SetupCertFullMipCount++;
        std::vector<std::vector<int>> yBestFull;
        bool optFull = false;
        const double valInc = val;
        const std::vector<std::vector<int>> yInc = yCand;
        auto t2 = std::chrono::steady_clock::now();
        double valFull = SolveTauJointSubproblem_WarmStart(
            env, pbList, zMask, &yCand, &optFull, &yBestFull
        );
        double dtFull = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        if (stats) {
            stats->jointSolved += 1;
            stats->timeJoint += dtFull;
        }
        if (std::isfinite(valFull)) {
            val = valFull;
            if (IsValidYPlan(yBestFull)) {
                yCand.swap(yBestFull);
            }
            exactAccepted = optFull;
            certAccepted = false;
            if (optFull) {
                certAccepted = true;
                certLB = val;
                hasFinalLB = true;
                finalLB = val;
            } else if (std::isfinite(certLB)) {
                hasFinalLB = true;
                finalLB = certLB;
            }
        } else {
            val = valInc;
            yCand = yInc;
        }
    }

    if (certAccepted && IsApproxCertWht() && !exactAccepted) {
        SetupCertApproxAcceptedCount++;
    }
}

static RLObjective EvaluateF_ByTau_RL(
    IloEnv env,
    int zMask,
    std::unordered_map<int, RLObjective>& cacheMask,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            return itMask->second;
        }
    }

    double totalLB = 0.0;
    double totalUB = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return {IloInfinity, IloInfinity};
        }
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) continue;
        RLObjective val = SolveTauRL(env, pbList, 30, 1.0);
        if (BBTimeLimitReached()) {
            return {IloInfinity, IloInfinity};
        }
        totalLB += val.lb;
        totalUB += val.ub;
    }

    RLObjective out{totalLB, totalUB};
    if (useCache) {
        cacheMask[zMask] = out;
    }
    return out;
}

static double EvaluateF_ByTau(
    IloEnv env,
    int zMask,
    int evalMode,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            return itMask->second;
        }
    }

    double total = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        int tauRep = tauReps[idx];
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            if (itTau != cacheTau.end()) {
                CacheTauHit++;
                total += itTau->second;
                continue;
            }
        }
        double val = 0.0;
        if (evalMode == 0) {
            val = SolveTauJointSubproblem(env, pbList, zMask);
            if (BBTimeLimitReached()) {
                return IloInfinity;
            }
        } else {
            for (int itItem = 0; itItem < N_it; ++itItem) {
                val += SolveTauItemSubproblem(env, itItem, pbList, zMask);
                if (BBTimeLimitReached()) {
                    return IloInfinity;
                }
            }
        }
        if (useCache) {
            cacheTau[key] = val;
        }
        total += val;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
    }
    return total;
}

static double EvaluateF_ByTau_BBCap(
    IloEnv env,
    int zMask,
    int evalMode,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            return itMask->second;
        }
    }

    double total = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        int tauRep = tauReps[idx];
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            if (itTau != cacheTau.end()) {
                CacheTauHit++;
                total += itTau->second;
                continue;
            }
        }
        double val = 0.0;
        if (evalMode == 0) {
            val = SolveTauJointSubproblem_BBCap(env, pbList, zMask, tauRep, key);
            if (BBTimeLimitReached()) {
                return IloInfinity;
            }
        } else {
            for (int itItem = 0; itItem < N_it; ++itItem) {
                val += SolveTauItemSubproblem(env, itItem, pbList, zMask);
                if (BBTimeLimitReached()) {
                    return IloInfinity;
                }
            }
        }
        if (useCache) {
            cacheTau[key] = val;
        }
        total += val;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
    }
    return total;
}

static double EvaluateF_ByTau_LP(
    IloEnv env,
    int zMask,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            return itMask->second;
        }
    }

    double total = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            if (itTau != cacheTau.end()) {
                total += itTau->second;
                continue;
            }
        }
        double val = SolveTauJointSubproblem_LP(env, pbList, zMask);
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        if (useCache) {
            cacheTau[key] = val;
        }
        total += val;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
    }
    return total;
}

static AirTauEvalResult EvaluateAIRTau(
    IloEnv env,
    const std::string& tauKey,
    const std::vector<int>& pbList,
    int zMask,
    uint64_t tauMask64,
    bool hasTauMask,
    EvalStats* stats,
    std::unordered_map<uint64_t, TauSubsetCacheEntry>* tauSubsetCache,
    std::vector<uint64_t>* tauSubsetMasks,
    long long* tauSubsetEpoch
) {
    AirTauEvalResult out;
    const bool trackTau = (hasTauMask && tauSubsetCache && tauSubsetMasks && tauSubsetEpoch);
    enum class AirUbSource {
        None,
        Reuse,
        Lagrangian,
        Seed
    };
    AirTauSolvedCount++;

    std::vector<std::vector<int>> yCand;
    bool haveY = false;
    bool fromReuse = false;
    AirUbSource ubSource = AirUbSource::None;
    uint64_t sourceMask64 = 0ULL;
    std::vector<uint64_t> sourceBitsDyn;
    double reuseCoverage = 0.0;
    std::vector<TauReuseRankedCandidate> reuseCands;

    if (ApproxReuseMode != 0) {
        if (N_pb <= 63) {
            if (hasTauMask) {
                fromReuse = CollectTopTauReuseY64(tauMask64, 3, reuseCands);
            }
        } else {
            fromReuse = CollectTopTauReuseYDyn(pbList, 3, reuseCands);
        }
    }

    if (fromReuse && !reuseCands.empty()) {
        const int candEvalCount = std::min(3, static_cast<int>(reuseCands.size()));
        double bestCandVal = IloInfinity;
        int bestCandIdx = -1;
        for (int c = 0; c < candEvalCount; ++c) {
            auto t0 = std::chrono::steady_clock::now();
            double candVal = SolveTauJointFixedY_LP(env, pbList, reuseCands[c].y);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            if (std::isfinite(candVal) && candVal < bestCandVal) {
                bestCandVal = candVal;
                bestCandIdx = c;
            }
        }
        if (bestCandIdx >= 0) {
            yCand = reuseCands[bestCandIdx].y;
            sourceMask64 = reuseCands[bestCandIdx].mask64;
            sourceBitsDyn = reuseCands[bestCandIdx].bits;
            reuseCoverage = reuseCands[bestCandIdx].coverage;
            out.val = bestCandVal;
            haveY = IsValidYPlan(yCand);
            if (haveY) {
                ubSource = AirUbSource::Reuse;
            }
        } else {
            fromReuse = false;
        }
    } else {
        fromReuse = false;
    }

    out.lb = ComputeTauApproxLb(env, tauKey, pbList, stats, false);
    if (ApproxLBMode == 1 && trackTau) {
        double packLb = ComputeTauPackingLbFromLowerBounds(
            tauMask64, true, *tauSubsetCache, *tauSubsetMasks
        );
        if (std::isfinite(packLb) && packLb > out.lb) {
            out.lb = packLb;
        }
    }

    bool accepted = AcceptApproxGap(out.lb, out.val);
    if (accepted) {
        AirTauAcceptedBaseCount++;
    }

    auto improveReuseUb = [&]() {
        if (ApproxLocalImproveMode == 0 || !fromReuse || !haveY || !IsValidYPlan(yCand)) {
            return;
        }

        std::vector<std::vector<char>> freeYLocal;
        std::vector<char> freeTLocal;
        bool useComponentFree = false;
        bool canIntensify = BuildConsensusFreeComponents(reuseCands, freeYLocal);
        if (canIntensify) {
            useComponentFree = true;
        }
        if (!canIntensify) {
            canIntensify = BuildDeltaFreePeriodsFromReuse(
                pbList,
                yCand,
                sourceMask64,
                (N_pb > 63 ? &sourceBitsDyn : nullptr),
                freeTLocal,
                &reuseCoverage
            );
        }
        if (!canIntensify) {
            return;
        }

        std::vector<std::vector<int>> yInt;
        const double miniMipTiLim = std::max(1.0, HeurIntensifyTiLim);
        auto t0 = std::chrono::steady_clock::now();
        double vInt = IloInfinity;
        if (useComponentFree) {
            std::vector<char> emptyFreeT;
            vInt = SolveTauJointPartialY_MIP(
                env, pbList, zMask, yCand, emptyFreeT, miniMipTiLim, &yInt, &freeYLocal
            );
        } else {
            vInt = SolveTauJointPartialY_MIP(
                env, pbList, zMask, yCand, freeTLocal, miniMipTiLim, &yInt
            );
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stats) {
            stats->jointSolved += 1;
            stats->timeJoint += dt;
        }
        if (std::isfinite(vInt) && vInt < out.val - 1e-9) {
            out.val = vInt;
            if (IsValidYPlan(yInt)) {
                yCand.swap(yInt);
                haveY = true;
                ubSource = AirUbSource::Reuse;
            }
        }
    };

    auto evaluateSeedUb = [&](const TauHeuristicSeed& seed) {
        if (!seed.valid || !IsValidYPlan(seed.y)) {
            return;
        }
        std::vector<std::vector<int>> ySeed;
        double vSeed = IloInfinity;
        if (ApproxLocalImproveMode != 0) {
            vSeed = SolveTauJointHeuristic(env, pbList, zMask, stats, &ySeed, &seed);
        } else {
            auto t0 = std::chrono::steady_clock::now();
            vSeed = SolveTauJointFixedY_LP(env, pbList, seed.y);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            ySeed = seed.y;
        }
        if (std::isfinite(vSeed) && vSeed < out.val - 1e-9) {
            out.val = vSeed;
            if (IsValidYPlan(ySeed)) {
                yCand = ySeed;
                haveY = true;
                ubSource = AirUbSource::Seed;
            }
        }
    };

    if (!accepted) {
        AirTauEnteredRefineCount++;
        if (ApproxGlobalLagLBMode != 0) {
            AirTauEnteredLagCount++;
            auto t0 = std::chrono::steady_clock::now();
            TauLagResult lagRes = SolveTauRLWithSeed(
                env,
                pbList,
                zMask,
                std::max(1, ApproxGlobalLagMaxIter),
                std::max(1e-6, ApproxGlobalLagStep0)
            );
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            if (std::isfinite(lagRes.lb)) {
                TauApproxGlobalLBCache[tauKey] = lagRes.lb;
                if (lagRes.lb > out.lb) {
                    out.lb = lagRes.lb;
                }
            }
            if (std::isfinite(lagRes.ub) && lagRes.ub < out.val - 1e-9) {
                out.val = lagRes.ub;
                if (lagRes.hasBestUbY && IsValidYPlan(lagRes.bestUbY)) {
                    yCand = lagRes.bestUbY;
                    haveY = true;
                    ubSource = AirUbSource::Lagrangian;
                }
            }
            accepted = AcceptApproxGap(out.lb, out.val);
            if (accepted) {
                AirTauAcceptedLocalCount++;
                AirTauAcceptedAfterLagCount++;
            } else {
                if (ubSource == AirUbSource::Reuse) {
                    improveReuseUb();
                    accepted = AcceptApproxGap(out.lb, out.val);
                    if (accepted) {
                        AirTauAcceptedLocalCount++;
                        AirTauAcceptedAfterReuseImproveCount++;
                    }
                } else {
                    evaluateSeedUb(lagRes.finalSeed);
                    accepted = AcceptApproxGap(out.lb, out.val);
                    if (accepted) {
                        AirTauAcceptedLocalCount++;
                        AirTauAcceptedAfterSeedImproveCount++;
                    }
                }
            }
        } else {
            if (ubSource == AirUbSource::Reuse) {
                improveReuseUb();
                accepted = AcceptApproxGap(out.lb, out.val);
                if (accepted) {
                    AirTauAcceptedLocalCount++;
                    AirTauAcceptedAfterReuseImproveCount++;
                }
            } else {
                TauHeuristicSeed baseSeed;
                std::vector<double> lambda0(N_tp + 1, 0.0);
                if (BuildTauHeuristicSeedFromLambda(env, pbList, zMask, lambda0, stats, true, baseSeed)) {
                    evaluateSeedUb(baseSeed);
                    accepted = AcceptApproxGap(out.lb, out.val);
                    if (accepted) {
                        AirTauAcceptedLocalCount++;
                        AirTauAcceptedAfterSeedImproveCount++;
                    }
                }
            }
        }
    }

    if (!accepted && ApproxExactFallbackMode != 0) {
        AirTauExactFallbackCount++;
        std::vector<std::vector<int>> yBest;
        bool optExact = false;
        bool targetHit = false;
        auto t0 = std::chrono::steady_clock::now();
        double vExact = SolveTauJointSubproblem_WarmStart(
            env, pbList, zMask, haveY ? &yCand : nullptr, &optExact, &yBest, out.lb, &targetHit
        );
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stats) {
            stats->jointSolved += 1;
            stats->timeJoint += dt;
        }
        if (std::isfinite(vExact)) {
            out.val = vExact;
            if (IsValidYPlan(yBest)) {
                yCand.swap(yBest);
                haveY = true;
            }
            out.exact = optExact;
            if (targetHit) {
                AirTauExactGapStoppedCount++;
            }
            if (optExact) {
                AirTauExactOptimalCount++;
                out.lb = std::max(out.lb, out.val);
            }
        }
    }

    out.hasY = haveY;
    if (haveY) {
        out.y = yCand;
    }

    if (std::isfinite(out.val) && out.hasY) {
        if (N_pb <= 63) {
            if (hasTauMask) {
                RegisterTauReuseY(tauMask64, out.y, out.val, out.exact);
            }
        } else {
            RegisterTauReuseY_Dyn(tauKey, pbList, out.y, out.val, out.exact);
        }
    }
    if (trackTau && std::isfinite(out.lb)) {
        RegisterTauSubsetLowerBound(
            tauMask64, out.lb, *tauSubsetCache, *tauSubsetMasks, *tauSubsetEpoch
        );
    }

    return out;
}

static double EvaluateF_ByTau_Heur(
    IloEnv env,
    int zMask,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    EvalStats* stats,
    bool useCache,
    CertAggStats* certAgg = nullptr,
    std::unordered_map<uint64_t, TauSubsetCacheEntry>* tauSubsetCache = nullptr,
    std::vector<uint64_t>* tauSubsetMasks = nullptr,
    long long* tauSubsetEpoch = nullptr
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        if (itMask != cacheMask.end()) {
            CacheMaskHit++;
            AccumulateMaskCertAggFromCache(zMask, cacheTau, certAgg);
            return itMask->second;
        }
    }

    double total = 0.0;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        uint64_t tauMask64 = 0ULL;
        bool hasTauMask = PbListToMask64(pbList, tauMask64);
        bool trackTau = (hasTauMask && tauSubsetCache && tauSubsetMasks && tauSubsetEpoch);
        if (stats) {
            stats->tauTotal += 1;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            if (itTau != cacheTau.end()) {
                CacheTauHit++;
                if (certAgg && SetupCertActive()) {
                    double certUb = IloInfinity;
                    double certLb = -IloInfinity;
                    if (GetStoredTauCertPair(key, itTau->second, certUb, certLb)) {
                        certAgg->finiteCount++;
                        certAgg->ubSum += certUb;
                        certAgg->lbSum += certLb;
                    }
                }
                if (trackTau) {
                    RegisterTauSubsetValue(tauMask64, itTau->second, false,
                                           *tauSubsetCache, *tauSubsetMasks, *tauSubsetEpoch);
                }
                total += itTau->second;
                continue;
            }
        }
        if (stats) {
            stats->tauSolved += 1;
        }
        bool hasCertPairLocal = false;
        double certPairUb = IloInfinity;
        double certPairLb = -IloInfinity;
        double val = IloInfinity;
        if (IsAIRWht()) {
            AirTauEvalResult air = EvaluateAIRTau(
                env, key, pbList, zMask, tauMask64, hasTauMask, stats,
                tauSubsetCache, tauSubsetMasks, tauSubsetEpoch
            );
            val = air.val;
        } else if (IsNAReuseWht()) {
            std::vector<std::vector<int>> yCand;
            bool fromReuse = false;
            uint64_t sourceMask64 = 0ULL;
            std::vector<uint64_t> sourceBitsDyn;
            double reuseCoverage = 0.0;
            std::vector<TauReuseRankedCandidate> reuseCands;
            if (N_pb <= 63) {
                if (hasTauMask) {
                    fromReuse = CollectTopTauReuseY64(tauMask64, 3, reuseCands);
                }
            } else {
                fromReuse = CollectTopTauReuseYDyn(pbList, 3, reuseCands);
            }
            if (fromReuse && !reuseCands.empty()) {
                const int candEvalCount = std::min(3, static_cast<int>(reuseCands.size()));
                double bestCandVal = IloInfinity;
                int bestCandIdx = -1;
                for (int c = 0; c < candEvalCount; ++c) {
                    auto t0 = std::chrono::steady_clock::now();
                    double candVal = SolveTauJointFixedY_LP(env, pbList, reuseCands[c].y);
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (stats) {
                        stats->jointSolved += 1;
                        stats->timeJoint += dt;
                    }
                    if (std::isfinite(candVal) && candVal < bestCandVal) {
                        bestCandVal = candVal;
                        bestCandIdx = c;
                    }
                }
                if (bestCandIdx >= 0) {
                    yCand = reuseCands[bestCandIdx].y;
                    sourceMask64 = reuseCands[bestCandIdx].mask64;
                    sourceBitsDyn = reuseCands[bestCandIdx].bits;
                    reuseCoverage = reuseCands[bestCandIdx].coverage;
                    val = bestCandVal;
                } else {
                    fromReuse = false;
                }
            } else {
                fromReuse = false;
            }
            if (fromReuse) {
                const bool certActive = SetupCertActive();
                const bool allowMiniMip = (!certActive || ActiveWht == 27 || ActiveWht == 28 || ActiveWht == 29);
                if (allowMiniMip && std::isfinite(val)) {
                    std::vector<std::vector<char>> freeYLocal;
                    std::vector<char> freeTLocal;
                    bool useComponentFree = false;
                    bool canIntensify = BuildConsensusFreeComponents(reuseCands, freeYLocal);
                    if (canIntensify) {
                        useComponentFree = true;
                    }
                    if (!canIntensify) {
                        canIntensify = BuildDeltaFreePeriodsFromReuse(
                            pbList,
                            yCand,
                            sourceMask64,
                            (N_pb > 63 ? &sourceBitsDyn : nullptr),
                            freeTLocal,
                            &reuseCoverage
                        );
                    }
                    if (canIntensify) {
                        std::vector<std::vector<int>> yInt;
                        const double miniMipTiLim = 10.0;
                        auto t1 = std::chrono::steady_clock::now();
                        double vInt = IloInfinity;
                        if (useComponentFree) {
                            std::vector<char> emptyFreeT;
                            vInt = SolveTauJointPartialY_MIP(
                                env, pbList, zMask, yCand, emptyFreeT, miniMipTiLim, &yInt, &freeYLocal
                            );
                        } else {
                            vInt = SolveTauJointPartialY_MIP(
                                env, pbList, zMask, yCand, freeTLocal, miniMipTiLim, &yInt
                            );
                        }
                        double dtInt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                        if (stats) {
                            stats->jointSolved += 1;
                            stats->timeJoint += dtInt;
                        }
                        if (std::isfinite(vInt) && vInt < val - 1e-9) {
                            val = vInt;
                            if (IsValidYPlan(yInt)) {
                                yCand.swap(yInt);
                            }
                        }
                    }
                }

                bool exactAccepted = false;
                bool certAccepted = false;
                double certLB = -IloInfinity;
                double finalLB = -IloInfinity;
                bool hasFinalLB = false;
                FinalizeTauCandidateWithCertification(
                    env, key, pbList, zMask, yCand, val, stats,
                    certAccepted, exactAccepted, certLB, hasFinalLB, finalLB
                );
                if (hasFinalLB && std::isfinite(finalLB)) {
                    hasCertPairLocal = true;
                    certPairUb = val;
                    certPairLb = finalLB;
                }

                if (std::isfinite(val)) {
                    if (N_pb <= 63) {
                        if (hasTauMask) {
                            RegisterTauReuseY(
                                tauMask64, yCand, val, exactAccepted,
                                certLB, certAccepted && std::isfinite(certLB), exactAccepted
                            );
                        }
                    } else {
                        RegisterTauReuseY_Dyn(
                            key, pbList, yCand, val, exactAccepted,
                            certLB, certAccepted && std::isfinite(certLB), exactAccepted
                        );
                    }
                }
            }
            if (!std::isfinite(val)) {
                std::vector<std::vector<int>> yHeur;
                val = SolveTauJointHeuristic(env, pbList, zMask, stats, &yHeur);
                if (std::isfinite(val) && IsValidYPlan(yHeur)) {
                    yCand = yHeur;
                    bool exactAccepted = false;
                    bool certAccepted = false;
                    double certLB = -IloInfinity;
                    double finalLB = -IloInfinity;
                    bool hasFinalLB = false;
                    FinalizeTauCandidateWithCertification(
                        env, key, pbList, zMask, yCand, val, stats,
                        certAccepted, exactAccepted, certLB, hasFinalLB, finalLB
                    );
                    if (hasFinalLB && std::isfinite(finalLB)) {
                        hasCertPairLocal = true;
                        certPairUb = val;
                        certPairLb = finalLB;
                    }
                    if (N_pb <= 63) {
                        if (hasTauMask) {
                            RegisterTauReuseY(
                                tauMask64, yCand, val, exactAccepted,
                                certLB, certAccepted && std::isfinite(certLB), exactAccepted
                            );
                        }
                    } else {
                        RegisterTauReuseY_Dyn(
                            key, pbList, yCand, val, exactAccepted,
                            certLB, certAccepted && std::isfinite(certLB), exactAccepted
                        );
                    }
                }
            }
        } else {
            val = SolveTauJointHeuristic(env, pbList, zMask, stats);
        }
        if (BBTimeLimitReached()) {
            return IloInfinity;
        }
        if (certAgg && hasCertPairLocal) {
            certAgg->finiteCount++;
            certAgg->ubSum += certPairUb;
            certAgg->lbSum += certPairLb;
        }
        if (hasCertPairLocal && std::isfinite(certPairLb)) {
            TauFinalLBCache[key] = certPairLb;
        }
        if (useCache) {
            cacheTau[key] = val;
        }
        if (trackTau) {
            RegisterTauSubsetValue(tauMask64, val, false,
                                   *tauSubsetCache, *tauSubsetMasks, *tauSubsetEpoch);
        }
        total += val;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
    }
    return total;
}

static EvalWithOpt EvaluateF_ByTau_BBCap_WithOpt(
    IloEnv env,
    int zMask,
    int evalMode,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    std::unordered_map<int, char>& cacheMaskOpt,
    std::unordered_map<std::string, char>& cacheTauOpt,
    EvalStats* stats,
    bool useCache
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        auto itMaskOpt = cacheMaskOpt.find(zMask);
        if (itMask != cacheMask.end() && itMaskOpt != cacheMaskOpt.end()) {
            CacheMaskHit++;
            return {itMask->second, itMaskOpt->second != 0};
        }
    }

    double total = 0.0;
    bool allOpt = true;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return {IloInfinity, false};
        }
        int tauRep = tauReps[idx];
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        if (stats) {
            stats->tauTotal += 1;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            auto itTauOpt = cacheTauOpt.find(key);
            if (itTau != cacheTau.end() && itTauOpt != cacheTauOpt.end()) {
                CacheTauHit++;
                total += itTau->second;
                allOpt = allOpt && (itTauOpt->second != 0);
                continue;
            }
        }
        double val = 0.0;
        bool opt = true;
        if (stats) {
            stats->tauSolved += 1;
        }
        if (evalMode == 0) {
            auto t0 = std::chrono::steady_clock::now();
            val = SolveTauJointSubproblem_BBCap(env, pbList, zMask, tauRep, key, &opt);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stats) {
                stats->jointSolved += 1;
                stats->timeJoint += dt;
            }
            if (BBTimeLimitReached()) {
                return {IloInfinity, false};
            }
        } else {
            for (int itItem = 0; itItem < N_it; ++itItem) {
                auto t0 = std::chrono::steady_clock::now();
                val += SolveTauItemSubproblem(env, itItem, pbList, zMask);
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->itemSolved += 1;
                    stats->timeItem += dt;
                }
                if (BBTimeLimitReached()) {
                    return {IloInfinity, false};
                }
            }
            opt = false;
        }
        if (useCache) {
            cacheTau[key] = val;
            cacheTauOpt[key] = opt ? 1 : 0;
        }
        total += val;
        allOpt = allOpt && opt;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
        cacheMaskOpt[zMask] = allOpt ? 1 : 0;
    }
    return {total, allOpt};
}

static EvalWithOpt EvaluateF_ByTau_WithOpt(
    IloEnv env,
    int zMask,
    int evalMode,
    std::unordered_map<int, double>& cacheMask,
    std::unordered_map<std::string, double>& cacheTau,
    std::unordered_map<int, char>& cacheMaskOpt,
    std::unordered_map<std::string, char>& cacheTauOpt,
    EvalStats* stats,
    bool useCache,
    std::unordered_map<uint64_t, TauSubsetCacheEntry>* tauSubsetCache = nullptr,
    std::vector<uint64_t>* tauSubsetMasks = nullptr,
    long long* tauSubsetEpoch = nullptr
){
    if (useCache) {
        auto itMask = cacheMask.find(zMask);
        auto itMaskOpt = cacheMaskOpt.find(zMask);
        if (itMask != cacheMask.end() && itMaskOpt != cacheMaskOpt.end()) {
            CacheMaskHit++;
            return {itMask->second, itMaskOpt->second != 0};
        }
    }

    double total = 0.0;
    bool allOpt = true;
    EnsureTauCache(zMask);
    const std::vector<int>& tauReps = TauReps[zMask];
    for (size_t idx = 0; idx < tauReps.size(); ++idx) {
        if (BBTimeLimitReached()) {
            return {IloInfinity, false};
        }
        int tauRep = tauReps[idx];
        const std::vector<int>& pbList = TauPbLists[zMask][idx];
        if (pbList.empty()) {
            continue;
        }
        uint64_t tauMask64 = 0ULL;
        bool hasTauMask = PbListToMask64(pbList, tauMask64);
        bool trackTau = (hasTauMask && tauSubsetCache && tauSubsetMasks && tauSubsetEpoch);
        if (stats) {
            stats->tauTotal += 1;
        }
        const std::string& key = TauKeys[zMask][idx];
        if (useCache) {
            auto itTau = cacheTau.find(key);
            auto itTauOpt = cacheTauOpt.find(key);
            if (itTau != cacheTau.end() && itTauOpt != cacheTauOpt.end()) {
                CacheTauHit++;
                if (trackTau) {
                    RegisterTauSubsetValue(tauMask64, itTau->second, itTauOpt->second != 0,
                                           *tauSubsetCache, *tauSubsetMasks, *tauSubsetEpoch);
                }
                total += itTau->second;
                allOpt = allOpt && (itTauOpt->second != 0);
                continue;
            }
        }
        double val = 0.0;
        bool opt = true;
        if (stats) {
            stats->tauSolved += 1;
        }
        if (evalMode == 0) {
            if (IsNAReuseWht()) {
                std::vector<std::vector<int>> yStart;
                bool haveStart = false;
                bool fromReuse = false;
                if (N_pb <= 63) {
                    if (hasTauMask) {
                        fromReuse = SelectBestTauReuseY(tauMask64, yStart);
                    }
                } else {
                    fromReuse = SelectBestTauReuseY_Dyn(pbList, yStart);
                }
                if (fromReuse) {
                    auto t0 = std::chrono::steady_clock::now();
                    double ubReuse = SolveTauJointFixedY_LP(env, pbList, yStart);
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (stats) {
                        stats->jointSolved += 1;
                        stats->timeJoint += dt;
                    }
                    if (std::isfinite(ubReuse) && IsValidYPlan(yStart)) {
                        haveStart = true;
                        if (N_pb <= 63) {
                            if (hasTauMask) {
                                RegisterTauReuseY(tauMask64, yStart, ubReuse, false);
                            }
                        } else {
                            RegisterTauReuseY_Dyn(key, pbList, yStart, ubReuse, false);
                        }
                    } else {
                        yStart.clear();
                    }
                }
                if (!haveStart) {
                    std::vector<std::vector<int>> yHeur;
                    double ubHeur = SolveTauJointHeuristic(env, pbList, zMask, stats, &yHeur);
                    if (std::isfinite(ubHeur) && IsValidYPlan(yHeur)) {
                        yStart = yHeur;
                        haveStart = true;
                        if (N_pb <= 63) {
                            if (hasTauMask) {
                                RegisterTauReuseY(tauMask64, yStart, ubHeur, false);
                            }
                        } else {
                            RegisterTauReuseY_Dyn(key, pbList, yStart, ubHeur, false);
                        }
                    }
                }

                std::vector<std::vector<int>> yBest;
                auto t0 = std::chrono::steady_clock::now();
                val = SolveTauJointSubproblem_WarmStart(
                    env, pbList, zMask, haveStart ? &yStart : nullptr, &opt, &yBest
                );
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->jointSolved += 1;
                    stats->timeJoint += dt;
                }
                if (std::isfinite(val)) {
                    if (N_pb <= 63) {
                        if (hasTauMask) {
                            if (IsValidYPlan(yBest)) {
                                RegisterTauReuseY(tauMask64, yBest, val, opt);
                            } else if (haveStart && IsValidYPlan(yStart)) {
                                RegisterTauReuseY(tauMask64, yStart, val, opt);
                            }
                        }
                    } else {
                        if (IsValidYPlan(yBest)) {
                            RegisterTauReuseY_Dyn(key, pbList, yBest, val, opt);
                        } else if (haveStart && IsValidYPlan(yStart)) {
                            RegisterTauReuseY_Dyn(key, pbList, yStart, val, opt);
                        }
                    }
                }
            } else {
                auto t0 = std::chrono::steady_clock::now();
                val = SolveTauJointSubproblem(env, pbList, zMask, &opt);
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->jointSolved += 1;
                    stats->timeJoint += dt;
                }
            }
            if (BBTimeLimitReached()) {
                return {IloInfinity, false};
            }
        } else {
            for (int itItem = 0; itItem < N_it; ++itItem) {
                auto t0 = std::chrono::steady_clock::now();
                val += SolveTauItemSubproblem(env, itItem, pbList, zMask);
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (stats) {
                    stats->itemSolved += 1;
                    stats->timeItem += dt;
                }
                if (BBTimeLimitReached()) {
                    return {IloInfinity, false};
                }
            }
            opt = false;
        }
        if (useCache) {
            cacheTau[key] = val;
            cacheTauOpt[key] = opt ? 1 : 0;
        }
        if (trackTau) {
            RegisterTauSubsetValue(tauMask64, val, opt,
                                   *tauSubsetCache, *tauSubsetMasks, *tauSubsetEpoch);
        }
        total += val;
        allOpt = allOpt && opt;
    }
    total = ApplyValueFunctionLB_BB(env, zMask, total);
    if (useCache) {
        cacheMask[zMask] = total;
        cacheMaskOpt[zMask] = allOpt ? 1 : 0;
    }
    return {total, allOpt};
}

static bool FeasibleMultiNode(int s0_mask, const std::vector<int>& blocks) {
    for (int block : blocks) {
        if ((block & ~s0_mask) == 0) {
            return false;
        }
    }
    return true;
}

static int MinAlphaInBlock(int block, int s0_mask) {
    int best = -1;
    double bestVal = 1e100;
    for (int i = 0; i < N_pn; ++i) {
        if ((block >> i) & 1) {
            if ((s0_mask >> i) & 1) {
                continue;
            }
            if (alpha[i] < bestVal) {
                bestVal = alpha[i];
                best = i;
            }
        }
    }
    return best;
}

static int BuildHeuristicSetFromBlocks(const std::vector<int>& blocks, int s0_mask) {
    int mask = 0;
    for (int block : blocks) {
        int j = MinAlphaInBlock(block, s0_mask);
        if (j >= 0) {
            mask |= (1 << j);
        }
    }
    return mask;
}

// Branch-and-bound (Ma) with exact F(S) evaluation via tau-class subproblems.
double Branch_and_Bound_Ma(IloEnv env, IloNumArray Solution, int bbMode, int evalMode, int bbCache) {
    try {
        auto algoStart = std::chrono::steady_clock::now();
        const double timeLimitSec = std::max(1.0, Wht24TimeLimitSec);
        StartBBTimeControl(timeLimitSec);
        TauReuseYCache.clear();
        TauReuseYMasks.clear();
        TauReuseDynIndexByKey.clear();
        TauReuseDynEntries.clear();
        TauApproxGlobalLBCache.clear();
        TauRelaxedLPCache.clear();
        TauCertCompLagCache.clear();
        TauFinalLBCache.clear();
        const int allMask = (N_pn >= 31) ? -1 : ((1 << N_pn) - 1);
        const bool useCache = (bbCache != 0);
        int fixed1 = 0;
        int fixed0 = 0;
        for (int j = 0; j < N_pn; ++j) {
            if (FixZ.size() == static_cast<size_t>(N_pn) && FixZ[j] >= 0) {
                if (FixZ[j] == 1) fixed1 |= (1 << j);
                else fixed0 |= (1 << j);
            }
        }
        if ((fixed1 & fixed0) != 0) {
            cout << "Ma-style branch-and-bound: infeasible fixed Z" << endl;
            StopBBTimeControl();
            return IloInfinity;
        }
        std::unordered_map<int, double> cacheMask;
        std::unordered_map<std::string, double> cacheTau;
        std::unordered_map<int, char> cacheMaskOpt;
        std::unordered_map<std::string, char> cacheTauOpt;
        std::unordered_map<int, double> cacheMaskJoint;
        std::unordered_map<std::string, double> cacheTauJoint;
        std::unordered_map<int, double> cacheMaskHeur;
        std::unordered_map<std::string, double> cacheTauHeur;
        std::unordered_map<uint64_t, TauSubsetCacheEntry> tauSubsetCache;
        std::vector<uint64_t> tauSubsetMasks;
        std::unordered_map<uint64_t, TauPackingMemoEntry> tauPackingMemo;
        long long tauSubsetEpoch = 0;
        std::vector<BBNode> pool;
        const bool allowDynFix = (DynamicFix == 1 && evalMode == 0);
        const bool allowResidualRNodeFix = (FixMode == 2 || FixMode == 3);
        const int kExactEvery = 0;
        const bool useHeurLB = (evalMode == 1 && HeurLBMode == 1);
        const bool enableTauPackingLB =
            ((TauPackingLBMode == 1) || (IsAIRWht() && ApproxLBMode == 1)) &&
            N_pb > 0 && N_pb <= 63;
        const bool allowHeurPackingValues = (evalMode == 1 && HeurLBMode == 1);
        std::vector<std::vector<uint64_t>> residualRBase;
        std::vector<int> idxByAlpha;
        if (allowResidualRNodeFix) {
            BuildFixingRsets(residualRBase);
            idxByAlpha.resize(N_pn);
            std::iota(idxByAlpha.begin(), idxByAlpha.end(), 0);
            std::sort(idxByAlpha.begin(), idxByAlpha.end(),
                      [](int a, int b) { return alpha[a] < alpha[b]; });
        }
        auto maskFullyCachedHeur = [&](int mask) -> bool {
            if (!useCache) {
                return false;
            }
            auto itMask = cacheMaskHeur.find(mask);
            if (itMask != cacheMaskHeur.end()) {
                return true;
            }
            EnsureTauCache(mask);
            const std::vector<int>& tauRepsMask = TauReps[mask];
            for (size_t idx = 0; idx < tauRepsMask.size(); ++idx) {
                const std::vector<int>& pbListTau = TauPbLists[mask][idx];
                if (pbListTau.empty()) {
                    continue;
                }
                const std::string& key = TauKeys[mask][idx];
                if (cacheTauHeur.find(key) == cacheTauHeur.end()) {
                    return false;
                }
            }
            return true;
        };
        auto maskFullyCachedWithOpt = [&](int mask) -> bool {
            if (!useCache) {
                return false;
            }
            auto itMask = cacheMask.find(mask);
            auto itMaskOpt = cacheMaskOpt.find(mask);
            if (itMask != cacheMask.end() && itMaskOpt != cacheMaskOpt.end()) {
                return true;
            }
            EnsureTauCache(mask);
            const std::vector<int>& tauRepsMask = TauReps[mask];
            for (size_t idx = 0; idx < tauRepsMask.size(); ++idx) {
                const std::vector<int>& pbListTau = TauPbLists[mask][idx];
                if (pbListTau.empty()) {
                    continue;
                }
                const std::string& key = TauKeys[mask][idx];
                if (cacheTau.find(key) == cacheTau.end()) {
                    return false;
                }
                if (cacheTauOpt.find(key) == cacheTauOpt.end()) {
                    return false;
                }
            }
            return true;
        };
        auto nodeNeedsFreshTauSolve = [&](int s0Mask, const std::vector<int>& blocks) -> bool {
            if (!useCache) {
                return true;
            }
            if (bbMode == 0) {
                int sMaxLocal = allMask & ~s0Mask;
                if (evalMode == 1) {
                    if (!maskFullyCachedHeur(sMaxLocal)) {
                        return true;
                    }
                    if (!useHeurLB && !maskFullyCachedWithOpt(sMaxLocal)) {
                        return true;
                    }
                    return false;
                }
                return !maskFullyCachedWithOpt(sMaxLocal);
            }
            if (!FeasibleMultiNode(s0Mask, blocks)) {
                return false;
            }
            int sMaxLocal = allMask & ~s0Mask;
            int sHeurLocal = BuildHeuristicSetFromBlocks(blocks, s0Mask);
            if (evalMode == 1) {
                if (!maskFullyCachedHeur(sHeurLocal)) {
                    return true;
                }
                if (!useHeurLB && !maskFullyCachedWithOpt(sMaxLocal)) {
                    return true;
                }
                return false;
            }
            if (!maskFullyCachedWithOpt(sMaxLocal)) {
                return true;
            }
            if (!maskFullyCachedWithOpt(sHeurLocal)) {
                return true;
            }
            return false;
        };

        if (bbMode == 0) {
            BBNode root;
            root.s1_mask = fixed1;
            root.s0_mask = fixed0;
            root.depth = 0;
            pool.push_back(root);
        } else {
            BBNode left;
            left.s1_mask = fixed1;
            left.s0_mask = allMask | fixed0;
            left.depth = 0;
            BBNode right;
            right.s1_mask = fixed1;
            right.s0_mask = fixed0;
            right.depth = 0;
            int freeMask = allMask & ~(fixed0 | fixed1);
            if (freeMask != 0) {
                right.blocks.push_back(freeMask);
            }
            if (fixed1 == 0) {
                pool.push_back(left);
            }
            pool.push_back(right);
        }

        double bestUB = IloInfinity;
        double bestLB = IloInfinity;
        int bestMaskUB = -1;
        int iter = 0;
        int pruned = 0;
        int totalDynFixed0 = 0;
        int totalFixMode2Fixed0 = 0;
        int nodesWithFixMode2 = 0;
        int targetPruned = 0;
        int targetFixed = 0;
        int targetSeen = 0;
        size_t lastFixCacheSize = 0;
        size_t lastFixCacheOptSize = 0;

        while (!pool.empty()) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count();
            if (elapsed >= timeLimitSec || BBTimeLimitReached()) {
                cout << "Ma-style branch-and-bound time limit reached: " << elapsed
                     << "s bestLB=" << bestLB
                     << " bestUB=" << bestUB
                     << " nodes=" << pool.size()
                     << endl;
                break;
            }
            BBNode node = pool.back();
            pool.pop_back();
            iter++;
            auto nodeStart = std::chrono::steady_clock::now();
            int sMax = -1;
            int sHeur = -1;
            double fS_branch = IloInfinity;
            double lbBaseNode = IloInfinity;
            double lbPackNode = IloInfinity;
            double lbPackCandidate = -IloInfinity;
            double alphaLbAdd = 0.0;
            int s0_eff = node.s0_mask | fixed0;
            int s1_eff = node.s1_mask | fixed1;
            EvalStats statsNode;
            CertAggStats certAggNode;
            if (allowResidualRNodeFix) {
                bool needFreshTau = nodeNeedsFreshTauSolve(s0_eff, node.blocks);
                if (needFreshTau) {
                    int localAdd = 0;
                    int addedLocal = ApplyResidualRNodeFixing(residualRBase, idxByAlpha, s1_eff, s0_eff, localAdd);
                    if (addedLocal > 0) {
                        node.s0_mask |= localAdd;
                        s0_eff |= localAdd;
                        totalFixMode2Fixed0 += addedLocal;
                        nodesWithFixMode2++;
                    }
                }
            }
            if (allowDynFix) {
                int localAdd = 0;
                int addedLocal = ApplySubmodularFixingForSupersets(cacheMask, cacheMaskOpt, s1_eff, s0_eff, localAdd);
                if (addedLocal > 0) {
                    node.s0_mask |= localAdd;
                    s0_eff |= localAdd;
                    totalDynFixed0 += addedLocal;
                }
            }
            auto targetCompatible = [&](int s0, int s1) -> bool {
                if (DebugTargetMask < 0) return false;
                return ((s1 & ~DebugTargetMask) == 0) && ((s0 & DebugTargetMask) == 0);
            };
            if (targetCompatible(s0_eff, s1_eff)) {
                targetSeen++;
            }
            if (bbMode == 0) {
                if ((s0_eff & s1_eff) != 0) {
                    continue;
                }
                sMax = allMask & ~s0_eff;
                alphaLbAdd = AlphaMask(s1_eff);
                if (enableTauPackingLB) {
                    EnsureTauCache(sMax);
                    double packedF = 0.0;
                    bool packValid = true;
                    for (const auto& pbListTau : TauPbLists[sMax]) {
                        if (pbListTau.empty()) {
                            continue;
                        }
                        uint64_t tauTargetMask = 0ULL;
                        if (!PbListToMask64(pbListTau, tauTargetMask)) {
                            packValid = false;
                            break;
                        }
                        if (IsAIRWht()) {
                            packedF += ComputeTauPackingLbFromLowerBounds(
                                tauTargetMask, false, tauSubsetCache, tauSubsetMasks
                            );
                        } else {
                            packedF += GetTauPackingLb(
                                tauTargetMask,
                                allowHeurPackingValues,
                                false,
                                tauSubsetCache,
                                tauSubsetMasks,
                                tauPackingMemo,
                                tauSubsetEpoch
                            );
                        }
                    }
                    if (packValid && std::isfinite(packedF)) {
                        lbPackCandidate = packedF + alphaLbAdd;
                    }
                }
                if (std::isfinite(lbPackCandidate) && std::isfinite(bestUB) && lbPackCandidate >= bestUB - 1e-6) {
                    if (lbPackCandidate < bestLB) {
                        bestLB = lbPackCandidate;
                    }
                    pruned++;
                    if (targetCompatible(s0_eff, s1_eff)) {
                        targetPruned++;
                        cout << "DebugTarget pre-pruned: S1=[" << MaskToBits(s1_eff)
                             << "] S0=[" << MaskToBits(s0_eff)
                             << "] lbPack=" << lbPackCandidate
                             << " bestUB=" << bestUB << endl;
                    }
                    cout << "Ma-style branch-and-bound node " << iter
                         << " lbBase=NA"
                         << " lbPack=" << lbPackCandidate
                         << " lb=" << lbPackCandidate
                         << " ub=NA"
                         << " bestLB=" << bestLB
                         << " bestUB=" << bestUB
                         << " tauSolve=" << statsNode.tauSolved
                         << " itemSolve=" << statsNode.itemSolved
                         << " jointSolve=" << statsNode.jointSolved
                         << " tItem=" << statsNode.timeItem
                         << " tJoint=" << statsNode.timeJoint
                         << " nodes=" << pool.size()
                         << " mode=" << (bbMode == 0 ? "single" : "multi")
                         << " eval=" << (evalMode == 0 ? "joint" : "item")
                         << " sMax=[" << (sMax >= 0 ? MaskToBits(sMax) : std::string("NA")) << "]"
                         << " prePrune=packing"
                         << " nodeTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - nodeStart).count()
                         << " totalTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count()
                         << endl;
                    continue;
                }
                double ubVal = 0.0;
                double lb = 0.0;
                if (evalMode == 1) {
                    ubVal = EvaluateF_ByTau_Heur(
                        env, sMax, cacheMaskHeur, cacheTauHeur, &statsNode, useCache, &certAggNode,
                        enableTauPackingLB ? &tauSubsetCache : nullptr,
                        enableTauPackingLB ? &tauSubsetMasks : nullptr,
                        enableTauPackingLB ? &tauSubsetEpoch : nullptr
                    );
                    if (kExactEvery > 0 && (iter % kExactEvery) == 0) {
                        double exact = EvaluateF_ByTau(env, sMax, 0, cacheMaskJoint, cacheTauJoint, useCache);
                        if (exact < ubVal) {
                            ubVal = exact;
                        }
                    }
                    if (useHeurLB) {
                        lb = ubVal + AlphaMask(s1_eff);
                        fS_branch = ubVal;
                    } else {
                        EvalWithOpt fMax = EvaluateF_ByTau_WithOpt(
                            env, sMax, evalMode, cacheMask, cacheTau, cacheMaskOpt, cacheTauOpt, &statsNode, useCache,
                            enableTauPackingLB ? &tauSubsetCache : nullptr,
                            enableTauPackingLB ? &tauSubsetMasks : nullptr,
                            enableTauPackingLB ? &tauSubsetEpoch : nullptr
                        );
                        lb = fMax.val + AlphaMask(s1_eff);
                        fS_branch = fMax.val;
                    }
                } else {
                    EvalWithOpt fMax = EvaluateF_ByTau_WithOpt(
                        env, sMax, evalMode, cacheMask, cacheTau, cacheMaskOpt, cacheTauOpt, &statsNode, useCache,
                        enableTauPackingLB ? &tauSubsetCache : nullptr,
                        enableTauPackingLB ? &tauSubsetMasks : nullptr,
                        enableTauPackingLB ? &tauSubsetEpoch : nullptr
                    );
                    lb = fMax.val + AlphaMask(s1_eff);
                    ubVal = fMax.val;
                    fS_branch = ubVal;
                }
                double ub = ubVal + AlphaMask(sMax);
                node.lb = lb;
                node.ub = ub;
            } else {
                if (!FeasibleMultiNode(s0_eff, node.blocks)) {
                    continue;
                }
                sMax = allMask & ~s0_eff;
                double minAlphaBlocks = 0.0;
                for (int block : node.blocks) {
                    int j = MinAlphaInBlock(block, s0_eff);
                    if (j >= 0) {
                        minAlphaBlocks += alpha[j];
                    }
                }
                alphaLbAdd = minAlphaBlocks;
                sHeur = BuildHeuristicSetFromBlocks(node.blocks, s0_eff);
                if (enableTauPackingLB) {
                    EnsureTauCache(sMax);
                    double packedF = 0.0;
                    bool packValid = true;
                    for (const auto& pbListTau : TauPbLists[sMax]) {
                        if (pbListTau.empty()) {
                            continue;
                        }
                        uint64_t tauTargetMask = 0ULL;
                        if (!PbListToMask64(pbListTau, tauTargetMask)) {
                            packValid = false;
                            break;
                        }
                        if (IsAIRWht()) {
                            packedF += ComputeTauPackingLbFromLowerBounds(
                                tauTargetMask, false, tauSubsetCache, tauSubsetMasks
                            );
                        } else {
                            packedF += GetTauPackingLb(
                                tauTargetMask,
                                allowHeurPackingValues,
                                false,
                                tauSubsetCache,
                                tauSubsetMasks,
                                tauPackingMemo,
                                tauSubsetEpoch
                            );
                        }
                    }
                    if (packValid && std::isfinite(packedF)) {
                        lbPackCandidate = packedF + alphaLbAdd;
                    }
                }
                if (std::isfinite(lbPackCandidate) && std::isfinite(bestUB) && lbPackCandidate >= bestUB - 1e-6) {
                    if (lbPackCandidate < bestLB) {
                        bestLB = lbPackCandidate;
                    }
                    pruned++;
                    if (targetCompatible(s0_eff, s1_eff)) {
                        targetPruned++;
                        cout << "DebugTarget pre-pruned: S1=[" << MaskToBits(s1_eff)
                             << "] S0=[" << MaskToBits(s0_eff)
                             << "] lbPack=" << lbPackCandidate
                             << " bestUB=" << bestUB << endl;
                    }
                    cout << "Ma-style branch-and-bound node " << iter
                         << " lbBase=NA"
                         << " lbPack=" << lbPackCandidate
                         << " lb=" << lbPackCandidate
                         << " ub=NA"
                         << " bestLB=" << bestLB
                         << " bestUB=" << bestUB
                         << " tauSolve=" << statsNode.tauSolved
                         << " itemSolve=" << statsNode.itemSolved
                         << " jointSolve=" << statsNode.jointSolved
                         << " tItem=" << statsNode.timeItem
                         << " tJoint=" << statsNode.timeJoint
                         << " nodes=" << pool.size()
                         << " mode=" << (bbMode == 0 ? "single" : "multi")
                         << " eval=" << (evalMode == 0 ? "joint" : "item")
                         << " sMax=[" << (sMax >= 0 ? MaskToBits(sMax) : std::string("NA")) << "]"
                         << (sHeur >= 0 ? " sHeur=[" + MaskToBits(sHeur) + "]" : std::string(""))
                         << " prePrune=packing"
                         << " nodeTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - nodeStart).count()
                         << " totalTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count()
                         << endl;
                    continue;
                }
                double ubVal = 0.0;
                double lb = 0.0;
                if (evalMode == 1) {
                    ubVal = EvaluateF_ByTau_Heur(
                        env, sHeur, cacheMaskHeur, cacheTauHeur, &statsNode, useCache, &certAggNode,
                        enableTauPackingLB ? &tauSubsetCache : nullptr,
                        enableTauPackingLB ? &tauSubsetMasks : nullptr,
                        enableTauPackingLB ? &tauSubsetEpoch : nullptr
                    );
                    if (kExactEvery > 0 && (iter % kExactEvery) == 0) {
                        double exact = EvaluateF_ByTau(env, sHeur, 0, cacheMaskJoint, cacheTauJoint, useCache);
                        if (exact < ubVal) {
                            ubVal = exact;
                        }
                    }
                    if (useHeurLB) {
                        lb = ubVal + minAlphaBlocks;
                    } else {
                        EvalWithOpt fMax = EvaluateF_ByTau_WithOpt(
                            env, sMax, evalMode, cacheMask, cacheTau, cacheMaskOpt, cacheTauOpt, &statsNode, useCache,
                            enableTauPackingLB ? &tauSubsetCache : nullptr,
                            enableTauPackingLB ? &tauSubsetMasks : nullptr,
                            enableTauPackingLB ? &tauSubsetEpoch : nullptr
                        );
                        lb = fMax.val + minAlphaBlocks;
                    }
                } else {
                    EvalWithOpt fMax = EvaluateF_ByTau_WithOpt(
                        env, sMax, evalMode, cacheMask, cacheTau, cacheMaskOpt, cacheTauOpt, &statsNode, useCache,
                        enableTauPackingLB ? &tauSubsetCache : nullptr,
                        enableTauPackingLB ? &tauSubsetMasks : nullptr,
                        enableTauPackingLB ? &tauSubsetEpoch : nullptr
                    );
                    lb = fMax.val + minAlphaBlocks;
                    EvalWithOpt fHeur = EvaluateF_ByTau_WithOpt(
                        env, sHeur, evalMode, cacheMask, cacheTau, cacheMaskOpt, cacheTauOpt, &statsNode, useCache,
                        enableTauPackingLB ? &tauSubsetCache : nullptr,
                        enableTauPackingLB ? &tauSubsetMasks : nullptr,
                        enableTauPackingLB ? &tauSubsetEpoch : nullptr
                    );
                    ubVal = fHeur.val;
                }
                double ub = ubVal + AlphaMask(sHeur);
                node.lb = lb;
                node.ub = ub;
            }
            lbBaseNode = node.lb;
            if (std::isfinite(lbPackCandidate)) {
                lbPackNode = lbPackCandidate;
                if (lbPackNode > node.lb) {
                    node.lb = lbPackNode;
                }
            } else {
                lbPackNode = lbBaseNode;
            }

            if (BBTimeLimitReached()) {
                double elapsedNow = std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count();
                cout << "Ma-style branch-and-bound time limit reached during node evaluation: " << elapsedNow
                     << "s bestLB=" << bestLB
                     << " bestUB=" << bestUB
                     << " nodes=" << pool.size()
                     << endl;
                break;
            }

            if (node.ub < bestUB) {
                bestUB = node.ub;
                bestMaskUB = (sHeur >= 0) ? sHeur : sMax;
                if (certAggNode.finiteCount > 0) {
                    const double alphaBest = (sHeur >= 0) ? AlphaMask(sHeur) : AlphaMask(sMax);
                    SetupCertBestSolLBCount = certAggNode.finiteCount;
                    SetupCertBestSolUBSum = certAggNode.ubSum + alphaBest;
                    SetupCertBestSolLBSum = certAggNode.lbSum + alphaBest;
                } else {
                    SetupCertBestSolLBCount = 0;
                    SetupCertBestSolUBSum = 0.0;
                    SetupCertBestSolLBSum = 0.0;
                }
            }
            if (node.lb < bestLB) {
                bestLB = node.lb;
            }

            if (allowDynFix) {
                if (cacheMask.size() != lastFixCacheSize || cacheMaskOpt.size() != lastFixCacheOptSize) {
                    int addMask = 0;
                    int added = ApplySubmodularFixingForSupersets(cacheMask, cacheMaskOpt, fixed1, fixed0, addMask);
                    if (added > 0) {
                        if (DebugTargetMask >= 0) {
                            int conflict = addMask & DebugTargetMask;
                            if (conflict != 0) {
                                targetFixed++;
                                cout << "DebugTarget conflict: addMask=[" << MaskToBits(addMask)
                                     << "] target=[" << MaskToBits(DebugTargetMask)
                                     << "] at S1=[" << MaskToBits(fixed1) << "]" << endl;
                            }
                        }
                        fixed0 |= addMask;
                        totalDynFixed0 += added;
                        cout << "Global submodular fixing applied: fixed=" << added
                             << " (total fixed=" << totalDynFixed0 << ")." << endl;
                    }
                    lastFixCacheSize = cacheMask.size();
                    lastFixCacheOptSize = cacheMaskOpt.size();
                }
            }


            if (node.lb >= bestUB - 1e-6) {
                pruned++;
                if (targetCompatible(s0_eff, s1_eff)) {
                    targetPruned++;
                    cout << "DebugTarget pruned: S1=[" << MaskToBits(s1_eff)
                         << "] S0=[" << MaskToBits(s0_eff)
                         << "] lb=" << node.lb
                         << " bestUB=" << bestUB << endl;
                }
                cout << "Ma-style branch-and-bound node " << iter
                     << " lbBase=" << lbBaseNode
                     << " lbPack=" << lbPackNode
                     << " lb=" << node.lb
                     << " ub=" << node.ub
                     << " bestLB=" << bestLB
                     << " bestUB=" << bestUB
                     << " tauSolve=" << statsNode.tauSolved
                     << " itemSolve=" << statsNode.itemSolved
                     << " jointSolve=" << statsNode.jointSolved
                     << " tItem=" << statsNode.timeItem
                     << " tJoint=" << statsNode.timeJoint
                     << " nodes=" << pool.size()
                     << " mode=" << (bbMode == 0 ? "single" : "multi")
                     << " eval=" << (evalMode == 0 ? "joint" : "item")
                     << " sMax=[" << (sMax >= 0 ? MaskToBits(sMax) : std::string("NA")) << "]"
                     << (sHeur >= 0 ? " sHeur=[" + MaskToBits(sHeur) + "]" : std::string(""))
                     << " nodeTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - nodeStart).count()
                     << " totalTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count()
                     << endl;
                continue;
            }

            if (bbMode == 0) {
                int fixed = s0_eff | s1_eff;
                if (fixed == allMask) {
                    continue;
                }
                int j = ChooseBranchVar_Option1(s0_eff, s1_eff, sMax, fS_branch, cacheMask);
                if (j < 0) {
                    for (int i = 0; i < N_pn; ++i) {
                        if (((fixed >> i) & 1) == 0) {
                            j = i;
                            break;
                        }
                    }
                }
                if (j < 0) {
                    continue;
                }
                BBNode plus = node;
                plus.s1_mask |= (1 << j);
                plus.depth = node.depth + 1;
                BBNode minus = node;
                minus.s0_mask |= (1 << j);
                minus.depth = node.depth + 1;
                pool.push_back(plus);
                pool.push_back(minus);
            } else {
                if (node.blocks.empty()) {
                    continue;
                }
                int T = node.blocks[0];
                int j = -1;
                for (int i = 0; i < N_pn; ++i) {
                    if ((T >> i) & 1) {
                        j = i;
                        break;
                    }
                }
                if (j < 0) {
                    continue;
                }
                int K = (1 << j);
                int TmK = T & ~K;

                auto rebuildBlocks = [&](const std::vector<int>& blocks, int replaceT, int add1, int add2) {
                    std::vector<int> out;
                    for (int b : blocks) {
                        if (b != replaceT) {
                            out.push_back(b);
                        }
                    }
                    if (add1 != 0) out.push_back(add1);
                    if (add2 != 0) out.push_back(add2);
                    return out;
                };

                BBNode P1 = node;
                P1.s0_mask |= K;
                P1.blocks = rebuildBlocks(node.blocks, T, TmK, 0);
                P1.depth = node.depth + 1;

                BBNode P2 = node;
                P2.s0_mask |= TmK;
                P2.blocks = rebuildBlocks(node.blocks, T, K, 0);
                P2.depth = node.depth + 1;

                BBNode P3 = node;
                P3.blocks = rebuildBlocks(node.blocks, T, K, TmK);
                P3.depth = node.depth + 1;

                pool.push_back(P1);
                pool.push_back(P2);
                pool.push_back(P3);
            }

            cout << "Ma-style branch-and-bound node " << iter
                 << " lbBase=" << lbBaseNode
                 << " lbPack=" << lbPackNode
                 << " lb=" << node.lb
                 << " ub=" << node.ub
                 << " bestLB=" << bestLB
                 << " bestUB=" << bestUB
                 << " tauSolve=" << statsNode.tauSolved
                 << " itemSolve=" << statsNode.itemSolved
                 << " jointSolve=" << statsNode.jointSolved
                 << " tItem=" << statsNode.timeItem
                 << " tJoint=" << statsNode.timeJoint
                 << " nodes=" << pool.size()
                 << " mode=" << (bbMode == 0 ? "single" : "multi")
                 << " eval=" << (evalMode == 0 ? "joint" : "item")
                 << " sMax=[" << (sMax >= 0 ? MaskToBits(sMax) : std::string("NA")) << "]"
                 << (sHeur >= 0 ? " sHeur=[" + MaskToBits(sHeur) + "]" : std::string(""))
                 << " nodeTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - nodeStart).count()
                 << " totalTime=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - algoStart).count()
                 << endl;
        }

        if (evalMode == 1 && SetupCertActive() && bestMaskUB >= 0) {
            CertAggStats bestCertAgg;
            AccumulateMaskCertAggFromCache(bestMaskUB, cacheTauHeur, &bestCertAgg);
            if (bestCertAgg.finiteCount > 0) {
                const double alphaBest = AlphaMask(bestMaskUB);
                SetupCertBestSolLBCount = bestCertAgg.finiteCount;
                SetupCertBestSolUBSum = bestCertAgg.ubSum + alphaBest;
                SetupCertBestSolLBSum = bestCertAgg.lbSum + alphaBest;
            } else {
                SetupCertBestSolLBCount = 0;
                SetupCertBestSolUBSum = 0.0;
                SetupCertBestSolLBSum = 0.0;
            }
        }

        cout << "Ma-style branch-and-bound done bestLB=" << bestLB << " bestUB=" << bestUB
             << " pruned=" << pruned
             << " bestS=[" << (bestMaskUB >= 0 ? MaskToBits(bestMaskUB) : std::string("NA")) << "]"
             << " dynFixed=" << totalDynFixed0
             << " fix2Vars=" << totalFixMode2Fixed0
             << " fix2Nodes=" << nodesWithFixMode2
             << " dbgSeen=" << targetSeen
             << " dbgPruned=" << targetPruned
             << " dbgFixed=" << targetFixed
             << endl;
        BestZCount = (bestMaskUB >= 0) ? CountBits(bestMaskUB) : -1;
        BBNodesExplored = iter;
        BBDynFixedCount = totalDynFixed0;
        BBPruned = pruned;
        Solution.add(bestLB);
        Solution.add(bestUB);
        StopBBTimeControl();
        return bestUB;
    } catch (IloException& e) {
        StopBBTimeControl();
        cout << "CPLEX exception in Branch_and_Bound_Ma: " << e.getMessage() << endl;
        return IloInfinity;
    } catch (...) {
        StopBBTimeControl();
        cout << "Unknown exception in Branch_and_Bound_Ma" << endl;
        return IloInfinity;
    }
}

