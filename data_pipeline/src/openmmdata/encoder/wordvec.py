import numpy as np
from model2vec import StaticModel


class WordVec:
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-8M",
    ):
        self.model = StaticModel.from_pretrained(model_name)

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 1024,
        show_progress_bar: bool = True,
    ):
        col_emb = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        col_emb = np.nan_to_num(col_emb, nan=0.0, posinf=0.0, neginf=0.0)

        return col_emb
