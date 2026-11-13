# Controlled cross-annotator generalization

Training 4000, validation 1000, test 986 per annotator and seed.

Each trained checkpoint supplies both target-annotator evaluations. Checkpoint selection uses the training annotator's validation Macro-F1.

Annotator identities and human annotation provenance were supplied by the user. Source hashes are in experiment.json. Sampling manifests report class prevalence, identifiers and allocation. The overlap report documents exclusions.

## Measured performance

```csv
model,seed,condition,accuracy,macro_f1
hybrid,67,A_to_A,0.7393509127789046,0.7168804116567618
hybrid,67,A_to_B,0.7322515212981744,0.7135887066339975
hybrid,67,B_to_A,0.6926977687626775,0.6662747729392948
hybrid,67,B_to_B,0.8012170385395537,0.7756991967618863
hybrid,42,A_to_A,0.6957403651115619,0.6736856796675356
hybrid,42,A_to_B,0.6987829614604463,0.673389695263272
hybrid,42,B_to_A,0.678498985801217,0.6505950303162101
hybrid,42,B_to_B,0.8103448275862069,0.7926879256574922
hybrid,123,A_to_A,0.7332657200811359,0.714853578543661
hybrid,123,A_to_B,0.691683569979716,0.6685993467193191
hybrid,123,B_to_A,0.6643002028397565,0.6480119753748177
hybrid,123,B_to_B,0.7484787018255578,0.7291186026991535
bert_cnn_lstm,67,A_to_A,0.716024340770791,0.6952274337261807
bert_cnn_lstm,67,A_to_B,0.7190669371196755,0.7038758260088535
bert_cnn_lstm,67,B_to_A,0.6744421906693712,0.6453336116224905
bert_cnn_lstm,67,B_to_B,0.8225152129817445,0.8000986293689737
bert_cnn_lstm,42,A_to_A,0.704868154158215,0.6691272152258914
bert_cnn_lstm,42,A_to_B,0.7099391480730223,0.6679955387219684
bert_cnn_lstm,42,B_to_A,0.6835699797160243,0.6536073303554544
bert_cnn_lstm,42,B_to_B,0.8002028397565923,0.7778259691636928
bert_cnn_lstm,123,A_to_A,0.7342799188640974,0.7271719692855912
bert_cnn_lstm,123,A_to_B,0.6947261663286004,0.6800786628221108
bert_cnn_lstm,123,B_to_A,0.6937119675456389,0.6653667357560226
bert_cnn_lstm,123,B_to_B,0.8093306288032455,0.7782759394719752
lstm_cnn,67,A_to_A,0.6531440162271805,0.6330420777833944
lstm_cnn,67,A_to_B,0.6592292089249493,0.6547487195926208
lstm_cnn,67,B_to_A,0.6227180527383367,0.6118883373624366
lstm_cnn,67,B_to_B,0.7210953346855984,0.7039920932912173
lstm_cnn,42,A_to_A,0.6399594320486816,0.6021657270600773
lstm_cnn,42,A_to_B,0.6480730223123732,0.6162358961490239
lstm_cnn,42,B_to_A,0.6460446247464503,0.6001078904263464
lstm_cnn,42,B_to_B,0.6987829614604463,0.6634241065662995
lstm_cnn,123,A_to_A,0.6176470588235294,0.6071959607755355
lstm_cnn,123,A_to_B,0.5811359026369168,0.5726284605169067
lstm_cnn,123,B_to_A,0.6480730223123732,0.6299966403766365
lstm_cnn,123,B_to_B,0.7068965517241379,0.6831148706865608
roberta,67,A_to_A,0.7119675456389453,0.6802202207369649
roberta,67,A_to_B,0.7352941176470589,0.7149202665985886
roberta,67,B_to_A,0.6825557809330629,0.6572000961215582
roberta,67,B_to_B,0.7971602434077079,0.7742099708201403
roberta,42,A_to_A,0.7079107505070994,0.6651501379057165
roberta,42,A_to_B,0.7190669371196755,0.6774729091459931
roberta,42,B_to_A,0.6551724137931034,0.6393963262811874
roberta,42,B_to_B,0.8052738336713996,0.7860763429537146
roberta,123,A_to_A,0.7139959432048681,0.6991498835602085
roberta,123,A_to_B,0.6815415821501014,0.6674312280909138
roberta,123,B_to_A,0.6825557809330629,0.6500016643699339
roberta,123,B_to_B,0.7636916835699797,0.7355240404461448
```

## Within minus cross contrasts

```csv
model,seed,contrast,difference,p_raw,p_holm
hybrid,67,delta_A,0.05060563871746693,0.0024987506246876563,0.029985007496251874
hybrid,67,delta_B,0.06211049012788872,0.0009995002498750624,0.014992503748125937
hybrid,42,delta_A,0.023090649351325432,0.1794102948525737,0.8970514742628686
hybrid,42,delta_B,0.11929823039422027,0.0004997501249375312,0.01199400299850075
hybrid,123,delta_A,0.06684160316884336,0.0004997501249375312,0.01199400299850075
hybrid,123,delta_B,0.06051925597983443,0.0009995002498750624,0.014992503748125937
bert_cnn_lstm,67,delta_A,0.0498938221036902,0.0024987506246876563,0.029985007496251874
bert_cnn_lstm,67,delta_B,0.09622280336012023,0.0004997501249375312,0.01199400299850075
bert_cnn_lstm,42,delta_A,0.015519884870436984,0.3638180909545227,0.8970514742628686
bert_cnn_lstm,42,delta_B,0.10983043044172447,0.0004997501249375312,0.01199400299850075
bert_cnn_lstm,123,delta_A,0.06180523352956868,0.0009995002498750624,0.014992503748125937
bert_cnn_lstm,123,delta_B,0.09819727664986433,0.0004997501249375312,0.01199400299850075
lstm_cnn,67,delta_A,0.021153740420957745,0.18190904547726136,0.8970514742628686
lstm_cnn,67,delta_B,0.04924337369859644,0.0029985007496251873,0.029985007496251874
lstm_cnn,42,delta_A,0.002057836633730936,0.9065467266366817,0.9065467266366817
lstm_cnn,42,delta_B,0.04718821041727561,0.0074962518740629685,0.05997001499250375
lstm_cnn,123,delta_A,-0.022800679601100993,0.2158920539730135,0.8970514742628686
lstm_cnn,123,delta_B,0.1104864101696541,0.0004997501249375312,0.01199400299850075
roberta,67,delta_A,0.0230201246154067,0.10094952523738131,0.7066466766616691
roberta,67,delta_B,0.059289704221551665,0.0004997501249375312,0.01199400299850075
roberta,42,delta_A,0.025753811624529166,0.11244377811094453,0.7066466766616691
roberta,42,delta_B,0.10860343380772153,0.0004997501249375312,0.01199400299850075
roberta,123,delta_A,0.04914821919027457,0.005997001499250375,0.05397301349325337
roberta,123,delta_B,0.06809281235523101,0.0004997501249375312,0.01199400299850075
bert_cnn_lstm,42,delta_overall,0.06267515765608073,,
bert_cnn_lstm,67,delta_overall,0.07305831273190522,,
bert_cnn_lstm,123,delta_overall,0.0800012550897165,,
hybrid,42,delta_overall,0.07119443987277285,,
hybrid,67,delta_overall,0.05635806442267782,,
hybrid,123,delta_overall,0.06368042957433889,,
lstm_cnn,42,delta_overall,0.024623023525503274,,
lstm_cnn,67,delta_overall,0.03519855705977709,,
lstm_cnn,123,delta_overall,0.043842865284276555,,
roberta,42,delta_overall,0.06717862271612535,,
roberta,67,delta_overall,0.04115491441847918,,
roberta,123,delta_overall,0.05862051577275279,,
```

## Across-seed summaries

```csv
model,measure,mean,sd,seeds,ci95_low,ci95_high,interval_method
bert_cnn_lstm,delta_A,0.04240631350123195,0.024033943994447953,3,,,individual paired intervals reported separately
bert_cnn_lstm,delta_B,0.10141683681723634,0.007352962290906886,3,,,individual paired intervals reported separately
bert_cnn_lstm,delta_overall,0.07191157515923415,0.00871978602750043,3,0.05220084996378929,0.09150670647581773,crossed song-cluster and seed-block bootstrap
hybrid,delta_A,0.04684596374587857,0.02211646122020061,3,,,individual paired intervals reported separately
hybrid,delta_B,0.08064265883398114,0.03348616006673593,3,,,individual paired intervals reported separately
hybrid,delta_overall,0.06374431128992986,0.007418394016175992,3,0.04417408904744236,0.08182488315416299,crossed song-cluster and seed-block bootstrap
lstm_cnn,delta_A,0.00013696581786256287,0.022040078683782657,3,,,individual paired intervals reported separately
lstm_cnn,delta_B,0.06897266476184205,0.03596664034027504,3,,,individual paired intervals reported separately
lstm_cnn,delta_overall,0.034554815289852304,0.00962607822053357,3,0.015502982364844617,0.05555296628712263,crossed song-cluster and seed-block bootstrap
roberta,delta_A,0.03264071847673681,0.014361108801953572,3,,,individual paired intervals reported separately
roberta,delta_B,0.0786619834615014,0.026300979351151298,3,,,individual paired intervals reported separately
roberta,delta_overall,0.05565135096911911,0.013263495131887267,3,0.03521770705453406,0.07731498683004408,crossed song-cluster and seed-block bootstrap
bert_cnn_lstm,A_to_A_accuracy,0.7183908045977011,0.01484799949695589,3,,,
bert_cnn_lstm,A_to_A_macro_f1,0.6971755394125544,0.02907137260754509,3,,,
bert_cnn_lstm,A_to_B_accuracy,0.7079107505070995,0.012296506747446413,3,,,
bert_cnn_lstm,A_to_B_macro_f1,0.6839833425176441,0.01825605780531177,3,,,
bert_cnn_lstm,B_to_A_accuracy,0.6839080459770116,0.009639335652050877,3,,,
bert_cnn_lstm,B_to_A_macro_f1,0.6547692259113225,0.0100669765371454,3,,,
bert_cnn_lstm,B_to_B_accuracy,0.810682893847194,0.011217484802594354,3,,,
bert_cnn_lstm,B_to_B_macro_f1,0.7854001793348805,0.012731219241421189,3,,,
hybrid,A_to_A_accuracy,0.7227856659905342,0.02361871328560993,3,,,
hybrid,A_to_A_macro_f1,0.7018065566226528,0.024374470325747032,3,,,
hybrid,A_to_B_accuracy,0.7075726842461122,0.021665273793119798,3,,,
hybrid,A_to_B_macro_f1,0.6851925828721962,0.024708131117403272,3,,,
hybrid,B_to_A_accuracy,0.678498985801217,0.014198782961460488,3,,,
hybrid,B_to_A_macro_f1,0.6549605928767742,0.009883119248138063,3,,,
hybrid,B_to_B_accuracy,0.786680189317106,0.03339677182221725,3,,,
hybrid,B_to_B_macro_f1,0.7658352417061773,0.032912579849522235,3,,,
lstm_cnn,A_to_A_accuracy,0.6369168356997972,0.017943008126728326,3,,,
lstm_cnn,A_to_A_macro_f1,0.6141345885396691,0.01656640199120571,3,,,
lstm_cnn,A_to_B_accuracy,0.6294793779580798,0.042236640840975656,3,,,
lstm_cnn,A_to_B_macro_f1,0.6145376920861838,0.041086459575447536,3,,,
lstm_cnn,B_to_A_accuracy,0.6389452332657201,0.014089699786460247,3,,,
lstm_cnn,B_to_A_macro_f1,0.6139976227218065,0.015055602178032632,3,,,
lstm_cnn,B_to_B_accuracy,0.7089249492900609,0.011293639681196796,3,,,
lstm_cnn,B_to_B_macro_f1,0.6835103568480259,0.02028688477086143,3,,,
roberta,A_to_A_accuracy,0.711291413116971,0.0030984284617686514,3,,,
roberta,A_to_A_macro_f1,0.6815067474009632,0.017036344657391147,3,,,
roberta,A_to_B_accuracy,0.7119675456389452,0.02757054202650751,3,,,
roberta,A_to_B_macro_f1,0.6866081346118319,0.02502781358468289,3,,,
roberta,B_to_A_accuracy,0.6734279918864097,0.01580979438957396,3,,,
roberta,B_to_A_macro_f1,0.6488660289242265,0.008956048572410967,3,,,
roberta,B_to_B_accuracy,0.788708586883029,0.02204181651481139,3,,,
roberta,B_to_B_macro_f1,0.7652701180733333,0.026435289814094293,3,,,
```

Positive contrasts are consistent with a within-annotator generalization advantage. Negative contrasts favor cross-annotator training on that target collection.

Individual paired bootstrap intervals and two-sided paired permutation tests use 2000 resamples. Holm correction spans every selected architecture, seed and target annotator as one prespecified family. Delta-overall is descriptive, with a crossed bootstrap sharing song weights across seeds and resampling seed blocks. These intervals condition on fitted checkpoints and cannot fully represent retraining uncertainty. With three seeds they are exploratory; with one seed they reflect test-song uncertainty only.

Annotators labeled different song collections. Annotation conventions, class prevalence, song difficulty, genre/domain, dataset composition and optimization can all contribute. The experiment cannot establish causal annotator bias or rank annotation quality. Independent labels from both annotators on the same songs are required for that question.

Existing B data previously served as thesis holdouts. This experiment repartitions that pool under the requested protocol. Earlier pipeline choices may have been informed by those songs.

## Class prevalence

```csv
seed,annotator,split,label,count,proportion
67,A,pool_class_counts,Safe,2831,0.4729368526561978
67,A,pool_class_counts,Mild,2011,0.3359505512863348
67,A,pool_class_counts,Explicit,1144,0.19111259605746742
67,A,training,Safe,1892,0.473
67,A,training,Mild,1344,0.336
67,A,training,Explicit,764,0.191
67,A,validation,Safe,473,0.473
67,A,validation,Mild,336,0.336
67,A,validation,Explicit,191,0.191
67,A,test,Safe,466,0.4726166328600406
67,A,test,Mild,331,0.3356997971602434
67,A,test,Explicit,189,0.19168356997971603
67,A,unused,Safe,0,
67,A,unused,Mild,0,
67,A,unused,Explicit,0,
67,B,pool_class_counts,Safe,4603,0.4605302651325663
67,B,pool_class_counts,Mild,3233,0.32346173086543273
67,B,pool_class_counts,Explicit,2159,0.21600800400200101
67,B,training,Safe,1842,0.4605
67,B,training,Mild,1294,0.3235
67,B,training,Explicit,864,0.216
67,B,validation,Safe,461,0.461
67,B,validation,Mild,323,0.323
67,B,validation,Explicit,216,0.216
67,B,test,Safe,454,0.460446247464503
67,B,test,Mild,319,0.3235294117647059
67,B,test,Explicit,213,0.21602434077079108
67,B,unused,Safe,1846,0.46046395609877777
67,B,unused,Mild,1297,0.32352207533050636
67,B,unused,Explicit,866,0.2160139685707159
42,A,pool_class_counts,Safe,2831,0.4729368526561978
42,A,pool_class_counts,Mild,2011,0.3359505512863348
42,A,pool_class_counts,Explicit,1144,0.19111259605746742
42,A,training,Safe,1892,0.473
42,A,training,Mild,1344,0.336
42,A,training,Explicit,764,0.191
42,A,validation,Safe,473,0.473
42,A,validation,Mild,336,0.336
42,A,validation,Explicit,191,0.191
42,A,test,Safe,466,0.4726166328600406
42,A,test,Mild,331,0.3356997971602434
42,A,test,Explicit,189,0.19168356997971603
42,A,unused,Safe,0,
42,A,unused,Mild,0,
42,A,unused,Explicit,0,
42,B,pool_class_counts,Safe,4603,0.4605302651325663
42,B,pool_class_counts,Mild,3233,0.32346173086543273
42,B,pool_class_counts,Explicit,2159,0.21600800400200101
42,B,training,Safe,1842,0.4605
42,B,training,Mild,1294,0.3235
42,B,training,Explicit,864,0.216
42,B,validation,Safe,461,0.461
42,B,validation,Mild,323,0.323
42,B,validation,Explicit,216,0.216
42,B,test,Safe,454,0.460446247464503
42,B,test,Mild,319,0.3235294117647059
42,B,test,Explicit,213,0.21602434077079108
42,B,unused,Safe,1846,0.46046395609877777
42,B,unused,Mild,1297,0.32352207533050636
42,B,unused,Explicit,866,0.2160139685707159
123,A,pool_class_counts,Safe,2831,0.4729368526561978
123,A,pool_class_counts,Mild,2011,0.3359505512863348
123,A,pool_class_counts,Explicit,1144,0.19111259605746742
123,A,training,Safe,1892,0.473
123,A,training,Mild,1344,0.336
123,A,training,Explicit,764,0.191
123,A,validation,Safe,473,0.473
123,A,validation,Mild,336,0.336
123,A,validation,Explicit,191,0.191
123,A,test,Safe,466,0.4726166328600406
123,A,test,Mild,331,0.3356997971602434
123,A,test,Explicit,189,0.19168356997971603
123,A,unused,Safe,0,
123,A,unused,Mild,0,
123,A,unused,Explicit,0,
123,B,pool_class_counts,Safe,4603,0.4605302651325663
123,B,pool_class_counts,Mild,3233,0.32346173086543273
123,B,pool_class_counts,Explicit,2159,0.21600800400200101
123,B,training,Safe,1842,0.4605
123,B,training,Mild,1294,0.3235
123,B,training,Explicit,864,0.216
123,B,validation,Safe,461,0.461
123,B,validation,Mild,323,0.323
123,B,validation,Explicit,216,0.216
123,B,test,Safe,454,0.460446247464503
123,B,test,Mild,319,0.3235294117647059
123,B,test,Explicit,213,0.21602434077079108
123,B,unused,Safe,1846,0.46046395609877777
123,B,unused,Mild,1297,0.32352207533050636
123,B,unused,Explicit,866,0.2160139685707159
```