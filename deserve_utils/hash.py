import hashlib

import torch


class MerkleTree:
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.layers = [
            [hashlib.sha256(i.numpy().tobytes()).digest() for i in data.view(-1)]
        ]
        self.branches = list(data.shape)
        for current_shape in reversed(data.shape):
            last_nodes = self.layers[0]
            nodes = []
            for i in range(0, len(last_nodes), current_shape):
                nodes.append(
                    hashlib.sha256(b"".join(last_nodes[i : i + current_shape])).digest()
                )
            self.layers.insert(0, nodes)

    def get_root(self) -> bytes:
        return self.layers[0][0]

    def generate_membership_proof(
        self, indices: list[int]
    ) -> list[tuple[bytes, bytes]]:
        proof = []
        for layer, index in reversed(list(enumerate(indices))):
            if layer != 0:
                parent = indices[layer - 1]
            else:
                parent = 0
            span = (
                parent * self.branches[layer],
                (parent + 1) * self.branches[layer],
            )
            nodes = self.layers[layer + 1]
            lhs = b""
            rhs = b""
            on_lhs = True
            for i in range(span[0], span[1]):
                if i == index + span[0]:
                    on_lhs = False
                elif on_lhs:
                    lhs += nodes[i]
                else:
                    rhs += nodes[i]
            proof.append((lhs, rhs))
        return proof

    @staticmethod
    def verify_membership(
        data: torch.Tensor,
        proof: list[tuple[bytes, bytes]],
        root: bytes,
    ) -> bool:
        subroot: bytes = MerkleTree(data).get_root()
        for lhs, rhs in proof:
            subroot = hashlib.sha256(lhs + subroot + rhs).digest()
        return subroot == root


class BatchMerkleTree:
    def __init__(self, data: torch.Tensor, keep_last_dim: bool):
        self.original_data = data
        self.keep_last_dim = keep_last_dim
        if keep_last_dim:
            self.data = data.reshape(-1, data.shape[-1])
        else:
            self.data = data.reshape(-1)
        self.layers = [
            [hashlib.sha256(i.detach().numpy().tobytes()).digest() for i in self.data]
        ]
        while len(self.layers[0]) > 1:
            nodes = []
            for i in range(0, len(self.layers[0]), 2):
                if i + 1 < len(self.layers[0]):
                    nodes.append(
                        hashlib.sha256(
                            self.layers[0][i] + self.layers[0][i + 1]
                        ).digest()
                    )
                else:
                    nodes.append(hashlib.sha256(self.layers[0][i]).digest())
            self.layers.insert(0, nodes)

    def get_root(self) -> bytes:
        return self.layers[0][0]

    def generate_membership_proof(
        self, indices: list[int]
    ) -> list[tuple[bytes, bytes]]:
        new_index = 0
        for i, index in enumerate(indices):
            new_index = new_index * self.original_data.shape[i] + index
        proof = []
        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            if new_index == len(layer) - 1 and len(layer) % 2 == 1:
                proof.append((b"", b""))
            elif new_index % 2 == 1:
                proof.append((layer[new_index - 1], b""))
            else:
                proof.append((b"", layer[new_index + 1]))
            # print(layer[new_index].hex())
            new_index //= 2
        # print(self.layers[0][0].hex())
        return proof

    @staticmethod
    def verify_membership(
        data: torch.Tensor,
        proof: list[tuple[bytes, bytes]],
        root: bytes,
    ) -> bool:
        subroot = hashlib.sha256(data.view(-1).detach().numpy().tobytes()).digest()
        for lhs, rhs in proof:
            subroot = hashlib.sha256(lhs + subroot + rhs).digest()
        return subroot == root


if __name__ == "__main__":
    data = torch.rand((12823, 4096))
    tree = BatchMerkleTree(data, keep_last_dim=False)
    proof = tree.generate_membership_proof([13])
    print(BatchMerkleTree.verify_membership(data[13], proof, tree.get_root()))
