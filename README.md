# 物理情報ニューラルネットワーク (PINN) サンプルプログラム
burn フレームワークを使用して、物理情報ニューラルネットワーク（PINN）を構築し、 1次元の移流方程式を解くサンプルです。

clap クレートを利用して、コマンドラインからtrain（学習）とinfer（推論）の 機能を個別に実行できます。

## 学習

```
cargo run --release -- train
```

## 推論

```
cargo run --release -- infer
```

## ドキュメント

```
cargo doc
```
