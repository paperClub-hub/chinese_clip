#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @ Date: 2022-08-09 15:57
import os
import json
import torch
import numpy as np
import pandas as pd
from typing  import List, Dict, Union


class EarlyStopping:
	""" 监控 valid_loss, 保留best_model """
	def __init__(self, patience=3, verbose=False, delta=0,  trace_func=print):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.trace_func = trace_func

	def __call__(self, val_loss, args, epoch,steps, model, optimizer, save_path=None):
		score = -val_loss
		if self.best_score is None:
			self.best_score = score

		elif score < self.best_score + self.delta:
			self.counter += 1
			# self.trace_func(f'EarlyStopping Counter: {self.counter} / {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, args, epoch, steps, model, optimizer, save_path)
			self.counter = 0

		# print("score: ", score, 'self.best_score: ', self.best_score)

	def save_checkpoint(self, val_loss, args, epoch, steps, model, optimizer, save_path):
		''' 模型权重保存 '''
		if self.verbose:
			self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model {save_path}')

		save_weights(args.name, epoch,steps, model, optimizer, save_path)

		self.val_loss_min = val_loss


def save_weights(name, epoch, steps, model, optimizer, save_path):
	print(f"Save Model {save_path} ")

	if save_path is not None:
		torch.save(
			{
				"epoch": epoch,
				"step": steps,
				"name": name,
				"state_dict": model.state_dict(),
				"optimizer": optimizer.state_dict(),
			},
			save_path,
		)


def monitor(loss_path: str, save_dir):

	""" 监控训练 loss, 绘图 """

	if not os.path.exists(loss_path):
		return

	loss = json.load(open(loss_path, 'r'))
	loss_tdf = pd.Dataframe(loss.get('train'))
	loss_vdf = pd.Dataframe(loss.get('valid'))

	# loss_tdf['epoch'] = loss_tdf['epoch'].map(str) + "_" + loss_tdf['steps'].map(str)
	# loss_vdf['epoch'] = loss_vdf['epoch'].map(str) + "_" + loss_vdf['steps'].map(str)
	# loss_tdf.drop(columns=['steps'], inplace=True)
	# loss_vdf.drop(columns=['steps'], inplace=True)

	loss_tdf.to_csv(os.path.join(save_dir, 'train_loss.csv'), index=False)
	loss_vdf.to_csv(os.path.join(save_dir, 'test_loss.csv'), index=False)








# if __name__ == '__main__':
# 	esp = EarlyStopping(patience=2, verbose=True)
# 	vloss = [19, 12, 13, 10, 11, 11]
#
# 	for v in vloss:
# 		esp(v)
# 		if esp.early_stop:
# 			print("终止 ")
