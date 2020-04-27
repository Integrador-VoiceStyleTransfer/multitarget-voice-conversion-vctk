import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from utils import SingleDataset
from solver import Solver
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--load_model', default=False, action='store_true')
  parser.add_argument('-flag', default='train')
  parser.add_argument('-hps_path', default='/home/daniel/Documents/voice_conversion/vctk.json')
  parser.add_argument('-load_model_path', default='/home/daniel/Documents/programacion/multitarget-voice-conversion-vctk/model.pkl')
  parser.add_argument('-dataset_path', default='/home/daniel/Documents/voice_integrador/vctk_old/data.h5')
  parser.add_argument('-index_path', default='/home/daniel/Documents/voice_integrador/vctk_old/index.json')
  parser.add_argument('-output_model_path', default='/home/daniel/Documents/voice_conversion/output.pkl')

  args = parser.parse_args()
  hps = Hps()
  hps.load(args.hps_path)
  hps_tuple = hps.get_tuple()
  dataset = SingleDataset(args.dataset_path, args.index_path, seg_len=hps_tuple.seg_len)
  data_loader = DataLoader(dataset)

  solver = Solver(hps_tuple, data_loader)
  if args.load_model:
    solver.load_model(args.load_model_path)

  if args.load_model:
    solver.load_model(args.load_model_path)

  # Speaker Classifier
  # Congelar hasta conv9 7, dejar la conv8 y reempl
  layers = list(solver.SpeakerClassifier.children())
  layers.pop(7)
  for l in layers:
    params = l.parameters()
    for p in params:
      p.requires_grad = False
  solver.SpeakerClassifier.conv9 = nn.Conv1d(512//4, 3, kernel_size=16)

  # Decoder
  # Cambiar las capas de embedding 
  solver.Decoder.emb1 = nn.Embedding(3, 512)
  solver.Decoder.emb2 = nn.Embedding(3, 512)
  solver.Decoder.emb3 = nn.Embedding(3, 512)
  solver.Decoder.emb4 = nn.Embedding(3, 512)
  solver.Decoder.emb5 = nn.Embedding(3, 512)
  # Generator
  # Cambiar las capas de embedding
  solver.Generator.emb1 = nn.Embedding(3, 512)
  solver.Generator.emb2 = nn.Embedding(3, 512)
  solver.Generator.emb3 = nn.Embedding(3, 512)
  solver.Generator.emb4 = nn.Embedding(3, 512)
  solver.Generator.emb5 = nn.Embedding(3, 512)

  # Discriminator
  # Congelar hasta conv6, dejar conv7 y reemplazar conv_classif
  layers = list(solver.PatchDiscriminator.module.children())
  layers.pop(6)
  for l in layers:
    params = l.parameters()
    for p in params:
      p.requires_grad = False
  solver.PatchDiscriminator.module.conv_classify =  nn.Conv2d(32, 3, kernel_size=(17, 4))

  solver.train(args.output_model_path, args.flag, mode='pretrain_G')
  solver.train(args.output_model_path, args.flag, mode='pretrain_D')
  solver.train(args.output_model_path, args.flag, mode='train')
  solver.train(args.output_model_path, args.flag, mode='patchGAN')