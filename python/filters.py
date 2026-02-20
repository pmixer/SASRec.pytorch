import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def from_end(sequence, max_length):
    return sequence[-max_length:]


class KMeansFilteringV2:
    def __init__(self, idx_to_item_embedding, n_clusters, max_per_cluster=10):
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto'
        )
        self.kmeans.fit(idx_to_item_embedding)
        self.labels = self.kmeans.labels_
        self.max_per_cluster = max_per_cluster

    def unique_clusters(self, sequence_of_item_indices, max_length):
        """One item per cluster, recency-prioritized."""
        added = set()
        result_sequence = []

        for i in range(len(sequence_of_item_indices) - 1, -1, -1):
            item = sequence_of_item_indices[i]
            cluster = self.labels[item]

            if cluster not in added:
                result_sequence.append(item)
                added.add(cluster)

            if len(result_sequence) == max_length:
                break

        return result_sequence[::-1]

    def mixed_unique_and_recent(self, sequence_of_item_indices, max_length):
        """
        Keep most recent (max_length - num_unique_clusters) items,
        plus one per unique cluster from the prefix.
        """
        if len(sequence_of_item_indices) <= max_length:
            return sequence_of_item_indices

        clusters_in_sequence = {
            self.labels[item] for item in sequence_of_item_indices
        }
        num_unique_clusters = len(clusters_in_sequence)

        num_recent = max(0, max_length - num_unique_clusters)

        recent_part = sequence_of_item_indices[-num_recent:] if num_recent > 0 else []
        prefix_part = sequence_of_item_indices[:-num_recent] if num_recent > 0 else sequence_of_item_indices

        unique_from_prefix = self.unique_clusters(prefix_part, num_unique_clusters)

        result = unique_from_prefix + recent_part
        return result[-max_length:]

    def bounded_cluster_filtering(self, sequence_of_item_indices, max_length):
        """Allow up to max_per_cluster items per cluster. Recency-prioritized."""
        cluster_counts = defaultdict(int)
        result_sequence = []

        for i in range(len(sequence_of_item_indices) - 1, -1, -1):
            item = sequence_of_item_indices[i]
            cluster = self.labels[item]

            if cluster_counts[cluster] < self.max_per_cluster:
                result_sequence.append(item)
                cluster_counts[cluster] += 1

            if len(result_sequence) == max_length:
                break

        return result_sequence[::-1]


class EmbeddingDiversityFiltering:
    def __init__(self, idx_to_item_embedding, similarity_threshold=0.8):
        self.embeddings = normalize(idx_to_item_embedding)
        self.similarity_threshold = similarity_threshold

    def distance_threshold_filtering(self, sequence_of_item_indices, max_length):
        """
        Keep an item only if its cosine similarity to all previously
        selected items is below similarity_threshold. Recency-prioritized.
        """
        selected = []
        selected_embeddings = []

        for i in range(len(sequence_of_item_indices) - 1, -1, -1):
            item = sequence_of_item_indices[i]
            emb = self.embeddings[item]

            if len(selected_embeddings) == 0:
                selected.append(item)
                selected_embeddings.append(emb)
            else:
                sims = np.dot(selected_embeddings, emb)
                if np.max(sims) < self.similarity_threshold:
                    selected.append(item)
                    selected_embeddings.append(emb)

            if len(selected) == max_length:
                break

        return selected[::-1]


class FastMMRFiltering:
    def __init__(self, idx_to_item_embedding, lambda_recency=0.7):
        self.embeddings = normalize(idx_to_item_embedding)
        self.lambda_recency = lambda_recency

    def mmr_filtering(self, sequence_of_item_indices, max_length):
        """
        Select items by MMR: score = lambda * recency - (1 - lambda) * max_sim_to_selected.
        Restores chronological order after selection.
        """
        if len(sequence_of_item_indices) <= max_length:
            return sequence_of_item_indices

        seq = np.array(sequence_of_item_indices)
        seq_emb = self.embeddings[seq]
        L = len(seq)

        recency_scores = np.linspace(0, 1, L)
        sim_matrix = seq_emb @ seq_emb.T

        selected_mask = np.zeros(L, dtype=bool)
        max_sim_to_selected = np.zeros(L)
        selected_indices = []

        for _ in range(max_length):
            scores = (
                self.lambda_recency * recency_scores
                - (1 - self.lambda_recency) * max_sim_to_selected
            )
            scores[selected_mask] = -np.inf

            best_idx = np.argmax(scores)
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)

            max_sim_to_selected = np.maximum(
                max_sim_to_selected,
                sim_matrix[:, best_idx]
            )

        selected_indices.sort()
        return seq[selected_indices].tolist()


class FilterByDifficulty:
    def __init__(self, model, itemnum, k_percent):
        self.model = model
        self.itemnum = itemnum
        self.k_percent = k_percent
        self.maxlen = model.pos_emb.num_embeddings - 1
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def get_difficulty(self, sequence):
        n = len(sequence)
        if n <= 1:
            return np.zeros(n)

        maxlen = self.maxlen
        itemnum = self.itemnum

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = sequence[-1]
        idx = maxlen - 1
        ts = set(sequence)

        for i in reversed(sequence[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            t = np.random.randint(1, itemnum + 1)
            while t in ts:
                t = np.random.randint(1, itemnum + 1)
            neg[idx] = t
            nxt = i
            idx -= 1
            if idx == -1:
                break

        with torch.no_grad():
            pos_logits, neg_logits = self.model(
                np.array([0]), seq[np.newaxis], pos[np.newaxis], neg[np.newaxis]
            )

        device = pos_logits.device
        pos_loss = self.bce(pos_logits[0], torch.ones(maxlen, device=device))
        neg_loss = self.bce(neg_logits[0], torch.zeros(maxlen, device=device))
        per_pos_loss = (pos_loss + neg_loss).detach().cpu().numpy()

        difficulty = np.zeros(n)
        num_placed = min(n - 1, maxlen)
        # placed positions correspond to sequence[n-1-num_placed : n-1]
        for j_local in range(num_placed):
            seq_idx = n - 1 - num_placed + j_local
            arr_idx = maxlen - num_placed + j_local
            difficulty[seq_idx] = per_pos_loss[arr_idx]

        # items at the start not placed (only when len > maxlen+1) and the last
        # item both get the mean of the computed difficulties
        computed_mean = difficulty[n - 1 - num_placed:n - 1].mean()
        for j in range(n - 1 - num_placed):
            difficulty[j] = computed_mean
        difficulty[n - 1] = computed_mean

        return difficulty

    def filter_easiest_k_percent(self, sequence, max_length):
        difficulty = self.get_difficulty(sequence)
        threshold = np.percentile(difficulty, self.k_percent)
        mask = difficulty >= threshold
        filtered = [item for item, keep in zip(sequence, mask) if keep]
        return filtered[-max_length:]

    def filter_hardest_k_percent(self, sequence, max_length):
        difficulty = self.get_difficulty(sequence)
        threshold = np.percentile(difficulty, 100 - self.k_percent)
        mask = difficulty <= threshold
        filtered = [item for item, keep in zip(sequence, mask) if keep]
        return filtered[-max_length:]
