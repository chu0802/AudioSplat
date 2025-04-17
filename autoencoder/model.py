import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(1, len(encoder_hidden_dims)):
            if i == 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
             
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder, self.decoder)
    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

class DefaultOpenCLIPAutoencoder(Autoencoder):
    encode_hiddien_dims = [512, 256, 128, 64, 32, 3]
    decode_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
    
    def __init__(self):
        super().__init__(self.encode_hiddien_dims, self.decode_hidden_dims)
        
class DefaultCLIPAutoencoder(Autoencoder):
    encode_hiddien_dims = [768, 512, 256, 128, 64, 32, 3]
    decode_hidden_dims = [16, 32, 64, 128, 256, 256, 512, 768]

    def __init__(self):
        super().__init__(self.encode_hiddien_dims, self.decode_hidden_dims)

class DefaultAudioCLIPAutoencoder(Autoencoder):
    encode_hiddien_dims = [1024, 512, 256, 128, 64, 32, 3]
    decode_hidden_dims = [16, 32, 64, 128, 256, 256, 512, 1024]

    def __init__(self):
        super().__init__(self.encode_hiddien_dims, self.decode_hidden_dims)



MODEL_DICT = {
    "open_clip": DefaultOpenCLIPAutoencoder,
    "clip": DefaultCLIPAutoencoder,
    "audio_clip": DefaultAudioCLIPAutoencoder,
}

def get_model(model_name):
    return MODEL_DICT[model_name]()


if __name__ == "__main__":
    ae = DefaultCLIPAutoencoder()
    print(ae)
