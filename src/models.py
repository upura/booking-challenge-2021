import gensim
import numpy as np
import torch
from torch import nn


class BookingNN(nn.Module):
    def __init__(
        self,
        n_city_id,
        n_booker_country,
        n_device_class,
        n_affiliate_id,
        n_month_checkin,
        n_hotel_country,
        emb_dim=512,
        rnn_dim=512,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        rnn_dropout=0.3,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        city_w2v = gensim.models.KeyedVectors.load_word2vec_format("../input/w2v/w2v_city.bin", binary=True)
        city_vectors = np.array([city_w2v[str(idx)] if str(idx) in city_w2v.vocab.keys() else np.zeros(emb_dim) for idx in range(n_city_id)])
        city_weights = torch.FloatTensor(city_vectors)
        self.city_id_embedding = nn.Embedding.from_pretrained(city_weights)
        self.booker_country_embedding = nn.Embedding(n_booker_country, emb_dim)
        self.device_class_embedding = nn.Embedding(n_device_class, emb_dim)
        self.affiliate_id_embedding = nn.Embedding(n_affiliate_id, emb_dim)
        self.month_checkin_embedding = nn.Embedding(n_month_checkin, emb_dim)
        self.hotel_country_embedding = nn.Embedding(n_hotel_country, emb_dim)

        self.cate_proj = nn.Sequential(
            nn.Linear(emb_dim * 6, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
        )
        self.cont_emb = nn.Sequential(
            nn.Linear(7, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_city_id),
        )

        self.n_city_id = n_city_id
        self.n_booker_country = n_booker_country
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout

    def forward(
        self,
        city_id_tensor,
        booker_country_tensor,
        device_class_tensor,
        affiliate_id_tensor,
        month_checkin_tensor,
        num_checkin_tensor,
        days_stay_tensor,
        days_move_tensor,
        hotel_country_tensor,
        num_visit_drop_duplicates_tensor,
        num_visit_tensor,
        num_visit_same_city_tensor,
        num_stay_consecutively_tensor,
    ):
        city_id_embedding = self.city_id_embedding(city_id_tensor)
        booker_country_embedding = self.booker_country_embedding(booker_country_tensor)
        device_class_embedding = self.device_class_embedding(device_class_tensor)
        affiliate_id_embedding = self.affiliate_id_embedding(affiliate_id_tensor)
        month_checkin_embedding = self.month_checkin_embedding(month_checkin_tensor)
        hotel_country_embedding = self.hotel_country_embedding(hotel_country_tensor)
        num_checkin_feature = num_checkin_tensor.unsqueeze(2)
        days_stay_feature = days_stay_tensor.unsqueeze(2)
        days_move_feature = days_move_tensor.unsqueeze(2)
        num_visit_drop_duplicates_feature = num_visit_drop_duplicates_tensor.unsqueeze(2)
        num_visit_feature = num_visit_tensor.unsqueeze(2)
        num_visit_same_city_feature = num_visit_same_city_tensor.unsqueeze(2)
        num_stay_consecutively_feature = num_stay_consecutively_tensor.unsqueeze(2)

        cate_emb = torch.cat(
            [
                city_id_embedding,
                booker_country_embedding,
                device_class_embedding,
                affiliate_id_embedding,
                month_checkin_embedding,
                hotel_country_embedding,
            ],
            dim=2,
        )
        cate_emb = self.cate_proj(cate_emb)

        cont_emb = torch.cat(
            [
                num_checkin_feature,
                days_stay_feature,
                days_move_feature,
                num_visit_drop_duplicates_feature,
                num_visit_feature,
                num_visit_same_city_feature,
                num_stay_consecutively_feature,
            ],
            dim=2,
        )
        cont_emb = self.cont_emb(cont_emb)

        out_s = torch.cat([cate_emb, cont_emb], dim=2)

        out_s, _ = self.lstm(out_s)
        out_s = out_s[:, -1, :]  # extrast last value of sequence
        out_s = self.ffn(out_s)
        return out_s
