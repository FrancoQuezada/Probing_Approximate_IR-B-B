#ifndef IndINCLUDE
#define IndINCLUDE
#include "Header.h"
#endif
#include <cstdint>
#include <numeric>

static size_t PairIndex(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    // i < j
    size_t base = static_cast<size_t>(i) * (2 * n - i - 1) / 2;
    return base + static_cast<size_t>(j - i - 1);
}

static void BuildRsets(std::vector<std::vector<uint64_t>>& R) {
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
                    size_t idx = PairIndex(pb, pb2, N_pb);
                    R[j][idx >> 6] |= (1ULL << (idx & 63));
                }
            }
        }
    }
}

static inline bool RSubset(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~b[i]) != 0ULL) return false;
    }
    return true;
}

static inline bool RSubsetUnion2(const std::vector<uint64_t>& a,
                                 const std::vector<uint64_t>& u) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~u[i]) != 0ULL) return false;
    }
    return true;
}

static inline bool RSubsetUnion2Plus(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& u,
                                     const std::vector<uint64_t>& r) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~(u[i] | r[i])) != 0ULL) return false;
    }
    return true;
}

static inline bool RSubsetUnion2Plus2(const std::vector<uint64_t>& a,
                                      const std::vector<uint64_t>& u,
                                      const std::vector<uint64_t>& r1,
                                      const std::vector<uint64_t>& r2) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~(u[i] | r1[i] | r2[i])) != 0ULL) return false;
    }
    return true;
}

void RunFixingPropositions(IloEnv env){
    FixZ.assign(N_pn, -1);
    std::vector<std::vector<uint64_t>> R;
    BuildRsets(R);
    const int pnCount = N_pn;
    const size_t pnPairs = static_cast<size_t>(pnCount) * static_cast<size_t>(pnCount - 1) / 2;
    const size_t blocks = R.empty() ? 0 : R[0].size();
    std::vector<std::vector<uint64_t>> union2(pnPairs, std::vector<uint64_t>(blocks, 0ULL));
    for (int i = 0; i < pnCount; ++i) {
        for (int j = i + 1; j < pnCount; ++j) {
            size_t p = PairIndex(i, j, pnCount);
            for (size_t b = 0; b < blocks; ++b) {
                union2[p][b] = R[i][b] | R[j][b];
            }
        }
    }

    std::vector<int> fix3, fix4, fix5;
    std::vector<unsigned char> fixed(N_pn, 0);
    const bool allowRProps = (N_pb >= 2);
    const bool use345 = allowRProps && (FixMode == 1 || FixMode == 3);

    if (use345) {
        std::vector<int> idxByAlpha(N_pn);
        std::iota(idxByAlpha.begin(), idxByAlpha.end(), 0);
        std::sort(idxByAlpha.begin(), idxByAlpha.end(),
                  [](int a, int b) { return alpha[a] < alpha[b]; });

        // Proposition 3
        for (int k = 0; k < N_pn; ++k) {
            for (int idx : idxByAlpha) {
                if (idx == k) continue;
                if (alpha[idx] > alpha[k]) break;
                if (alpha[idx] <= alpha[k] && RSubset(R[k], R[idx])) {
                    fix3.push_back(k);
                    fixed[k] = 1;
                    break;
                }
            }
        }

        // Proposition 4
        for (int k = 0; k < N_pn; ++k) {
            if (fixed[k]) continue;
            bool fixedLocal = false;
            std::vector<int> candidates;
            candidates.reserve(N_pn);
            for (int idx : idxByAlpha) {
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
                    size_t p = PairIndex(i, j, pnCount);
                    if (RSubsetUnion2(R[k], union2[p])) {
                        fix4.push_back(k);
                        fixedLocal = true;
                    }
                }
            }
            if (fixedLocal) {
                fixed[k] = 1;
            }
        }

        // Proposition 5 (max union size = 4)
        for (int k = 0; k < N_pn; ++k) {
            if (fixed[k]) continue;
            bool fixedLocal = false;
            std::vector<int> candidates;
            candidates.reserve(N_pn);
            for (int idx : idxByAlpha) {
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
                    size_t p = PairIndex(i, j, pnCount);
                    for (int c = b + 1; c < csz && !fixedLocal; ++c) {
                        int l = candidates[c];
                        if (alpha[i] + alpha[j] + alpha[l] > alpha[k]) break;
                        if (RSubsetUnion2Plus(R[k], union2[p], R[l])) {
                            fix5.push_back(k);
                            fixedLocal = true;
                            break;
                        }
                        for (int d = c + 1; d < csz && !fixedLocal; ++d) {
                            int m = candidates[d];
                            if (alpha[i] + alpha[j] + alpha[l] + alpha[m] > alpha[k]) break;
                            if (RSubsetUnion2Plus2(R[k], union2[p], R[l], R[m])) {
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
    }

    for (int j : fix3) FixZ[j] = 0;
    for (int j : fix4) FixZ[j] = 0;
    for (int j : fix5) FixZ[j] = 0;
    FixCount = 0;
    for (int j = 0; j < N_pn; ++j) {
        if (FixZ[j] >= 0) {
            FixCount++;
        }
    }

    cout << "Prop3 fix count: " << fix3.size() << endl;
    if (!fix3.empty()) {
        cout << "Prop3 fix idx: [";
        for (size_t i = 0; i < fix3.size(); ++i) {
            cout << fix3[i];
            if (i + 1 < fix3.size()) cout << ",";
        }
        cout << "]" << endl;
    }
    cout << "Prop4 fix count: " << fix4.size() << endl;
    if (!fix4.empty()) {
        cout << "Prop4 fix idx: [";
        for (size_t i = 0; i < fix4.size(); ++i) {
            cout << fix4[i];
            if (i + 1 < fix4.size()) cout << ",";
        }
        cout << "]" << endl;
    }

    cout << "Prop5 fix count: " << fix5.size() << endl;
    if (!fix5.empty()) {
        cout << "Prop5 fix idx: [";
        for (size_t i = 0; i < fix5.size(); ++i) {
            cout << fix5[i];
            if (i + 1 < fix5.size()) cout << ",";
        }
        cout << "]" << endl;
    }
}
