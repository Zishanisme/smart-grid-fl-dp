"""
secagg_dh.py
============
Simulation of the Bonawitz et al. (2017) Secure Aggregation protocol.



             using their half of the shared secret

For production: replace with PyCryptodome ECDH + TLS channels.
"""

import hashlib
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Simplified DH group (proxy for ECDH — same algebraic structure)
# ---------------------------------------------------------------------------

# Safe prime p and generator g for a 256-bit DH group
# In production: use NIST P-256 curve via cryptography library
DH_PRIME = (
    0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1
    + 0x29024E088A67CC74020BBEA63B139B22514A08798E3404DD
    + 0xEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245
    + 0xE485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED
    + 0xEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D
    + 0xC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F
    + 0x83655D23DCA3AD961C62F356208552BB9ED529077096966D
    + 0x670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF
)
DH_GENERATOR = 2


class DHKeyPair:
    """
    Simulated Diffie-Hellman key pair.
    In production replace with:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    """
    def __init__(self, rng: np.random.Generator):
        # Private key: random integer in [2, p-2]
        self.private = int.from_bytes(rng.bytes(32), "big") % (DH_PRIME - 2) + 2
        # Public key: g^private mod p
        self.public  = pow(DH_GENERATOR, self.private, DH_PRIME)

    def shared_secret(self, other_public: int) -> bytes:
        """Compute shared secret: other_public^private mod p → 32 bytes."""
        secret_int = pow(other_public, self.private, DH_PRIME)
        return secret_int.to_bytes(64, "big")[:32]


def prg_masks(
    shared_secret: bytes,
    layer_shapes:  List[tuple],
    client_i:      int,
    client_j:      int,
    round_num:     int,
) -> List[np.ndarray]:
    """
    Deterministic PRG: expand shared secret into per-layer mask arrays.
    Uses SHAKE-256 (extendable output function) seeded by:
      shared_secret || round_num || (i, j ordering)

    Sign convention:
      client with lower index ADDS mask
      client with higher index SUBTRACTS mask
    → masks cancel exactly in the server sum
    """
    sign   = 1.0 if client_i < client_j else -1.0
    i_low  = min(client_i, client_j)
    i_high = max(client_i, client_j)

    seed = (shared_secret
            + struct.pack(">III", round_num, i_low, i_high))

    masks = []
    offset = 0
    total_elements = sum(int(np.prod(s)) for s in layer_shapes)

    # SHAKE-256: arbitrary output length
    shake = hashlib.shake_256(seed)
    # 4 bytes per float32
    raw   = shake.digest(total_elements * 4)

    for shape in layer_shapes:
        n = int(np.prod(shape))
        chunk = np.frombuffer(raw[offset * 4:(offset + n) * 4], dtype=np.uint32)
        # Map uint32 → float32 in [-0.01, 0.01] (small enough not to dominate signal)
        mask = (chunk.astype(np.float32) / np.iinfo(np.uint32).max - 0.5) * 0.02
        masks.append(sign * mask.reshape(shape))
        offset += n

    return masks


# ---------------------------------------------------------------------------
# Secure Aggregator — Bonawitz protocol
# ---------------------------------------------------------------------------

class BonawitzSecAgg:
    """
    Simulation of the Bonawitz et al. (2017) SecAgg protocol.

    Usage:
        secagg = BonawitzSecAgg(n_clients=3, dropout_threshold=2)
        secagg.setup(round_num=1)                    # key exchange
        masked = secagg.mask_update(i, update, round_num)  # per client
        aggregated = secagg.aggregate(masked_updates, sizes, round_num)

    
    """

    def __init__(
        self,
        n_clients:         int,
        dropout_threshold: int  = None,  # min surviving clients
        seed:              int  = 0,
    ):
        self.n          = n_clients
        self.threshold  = dropout_threshold or max(2, n_clients - 1)
        self.rng        = np.random.default_rng(seed)

        # Generate DH key pairs (one per client)
        self.keypairs: List[DHKeyPair] = [
            DHKeyPair(np.random.default_rng(seed + i))
            for i in range(n_clients)
        ]

        # Pre-compute pairwise shared secrets (simulates Round 1 key exchange)
        self.shared_secrets: Dict[Tuple[int,int], bytes] = {}
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                secret = self.keypairs[i].shared_secret(self.keypairs[j].public)
                self.shared_secrets[(i, j)] = secret
                self.shared_secrets[(j, i)] = secret  # symmetric

        print(f"[BonawitzSecAgg] {n_clients} clients | "
              f"dropout threshold={self.threshold} | "
              f"{n_clients*(n_clients-1)//2} pairwise secrets established")
        print(f"  NOTE: This is an algorithmic simulation of the Bonawitz 2017 protocol.")
        print(f"  Cryptographic primitives are simplified (modular DH, not ECDH).")
        print(f"  For production: replace DHKeyPair with X25519 via cryptography library.")

    def mask_update(
        self,
        client_id:     int,
        update:        List[np.ndarray],
        round_num:     int,
        dropped_ids:   List[int] = None,
    ) -> List[np.ndarray]:
        """
        Client i masks its model update.
        Returns masked update = w_i + Σ_j mask(i,j)
        Masks from dropped clients are excluded.
        """
        dropped = set(dropped_ids or [])
        layer_shapes = [u.shape for u in update]
        masked = [u.copy() for u in update]

        for j in range(self.n):
            if j == client_id or j in dropped:
                continue
            secret = self.shared_secrets[(client_id, j)]
            masks  = prg_masks(secret, layer_shapes, client_id, j, round_num)
            for l in range(len(update)):
                masked[l] = masked[l] + masks[l]

        return masked

    def aggregate(
        self,
        masked_updates: List[Optional[List[np.ndarray]]],  # None = dropped client
        dataset_sizes:  List[int],
        round_num:      int,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Server aggregates masked updates.

        

        Returns (aggregated_weights, aggregation_report).
        """
        surviving     = [i for i, u in enumerate(masked_updates) if u is not None]
        dropped       = [i for i, u in enumerate(masked_updates) if u is None]
        n_surviving   = len(surviving)

        if n_surviving < self.threshold:
            raise RuntimeError(
                f"Only {n_surviving} clients survived, need ≥ {self.threshold}. "
                f"Aborting round for privacy safety."
            )

        if dropped:
            print(f"  [SecAgg] Round {round_num}: {len(dropped)} client(s) dropped "
                  f"({dropped}). Proceeding with {n_surviving}/{self.n}.")

        # Weighted sum of surviving masked updates
        # active_sizes = [dataset_sizes[i] for i in surviving]
        # total        = sum(active_sizes)
        n_layers     = len(masked_updates[surviving[0]])

        aggregated = []
        for l in range(n_layers):
            layer = sum(
                masked_updates[i][l]
                for i in surviving
            )
            aggregated.append(layer)

        report = {
            "round":         round_num,
            "n_surviving":   n_surviving,
            "n_dropped":     len(dropped),
            "dropped_ids":   dropped,
            "secagg_active": True,
            "protocol":      "Bonawitz et al. 2017 (algorithmic simulation)",
        }
        return aggregated, report
