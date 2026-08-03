"""3-population admixed+related fixture -> PLINK for GENESIS PC-AiR validation.
3 source pops (Balding-Nichols, Fst) => TWO real ancestry axes (PC1, PC2),
plus parent-offspring pairs for relatedness. Deterministic."""
import numpy as np
from pathlib import Path
from export_fixture import write_plink_bed

def make_3pop(n_per_pop=45, n_related=12, m=2500, fst=0.15, seed=7):
    rng = np.random.default_rng(seed)
    K=3; n=K*n_per_pop
    p_anc = 0.1 + 0.8*rng.random(m)
    a=p_anc*(1-fst)/fst; b=(1-p_anc)*(1-fst)/fst
    pops=[rng.beta(a,b) for _ in range(K)]           # 3 pop-specific freq vectors
    pop_label=np.repeat(np.arange(K), n_per_pop)
    G=np.zeros((n,m))
    for i in range(n):
        G[i]=rng.binomial(2, pops[pop_label[i]])
    # parent-offspring within pop 0 and pop 1
    related=[]
    for r in range(n_related):
        parent=r; child=n_per_pop-1-r
        if child<=parent: break
        transmit=np.where(G[parent]==1, rng.binomial(1,0.5,m), (G[parent]==2).astype(float))
        other=rng.binomial(1, pops[0])
        G[child]=np.clip(transmit+other,0,2)
        related.append((parent,child))
    return G.astype(np.float64), pop_label, related

G,pop,related=make_3pop()
n,m=G.shape
ids=[f"IND{i:04d}" for i in range(n)]; snps=[f"snp{j}" for j in range(m)]
out=Path("out3")
write_plink_bed(G, out/"fixture", ids, snps)
np.save(out/"G.npy", G)
np.savez(out/"meta.npz", sample_ids=ids, pop_label=pop, related_pairs=np.array(related))
print(f"3-pop fixture: n={n} m={m} pops=3 related={len(related)} -> out3/")
