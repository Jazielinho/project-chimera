## Model summary (torchinfo)

```
============================================================================================================================================================================================================================
Layer (type (var_name))                                                Kernel Shape              Input Shape               Output Shape              Param #                   Mult-Adds                 Trainable
============================================================================================================================================================================================================================
TextEncoder (TextEncoder)                                              --                        [1, 32]                   [1, 256]                  --                        --                        Partial
├─BertModel (model)                                                    --                        --                        [1, 384]                  --                        --                        False
│    └─BertEmbeddings (embeddings)                                     --                        --                        [1, 32, 384]              --                        --                        False
│    │    └─Embedding (word_embeddings)                                --                        [1, 32]                   [1, 32, 384]              (11,720,448)              11,720,448                False
│    │    └─Embedding (token_type_embeddings)                          --                        [1, 32]                   [1, 32, 384]              (768)                     768                       False
│    │    └─Embedding (position_embeddings)                            --                        [1, 32]                   [1, 32, 384]              (196,608)                 196,608                   False
│    │    └─LayerNorm (LayerNorm)                                      --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    └─Dropout (dropout)                                          --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    └─BertEncoder (encoder)                                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    └─ModuleList (layer)                                         --                        --                        --                        --                        --                        False
│    │    │    └─BertLayer (0)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    │    └─BertLayer (1)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    │    └─BertLayer (2)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    │    └─BertLayer (3)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    │    └─BertLayer (4)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    │    │    └─BertLayer (5)                                         --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    └─BertAttention (attention)                        --                        [1, 32, 384]              [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─BertSdpaSelfAttention (self)                --                        [1, 32, 384]              [1, 32, 384]              (443,520)                 443,520                   False
│    │    │    │    │    └─BertSelfOutput (output)                     --                        [1, 32, 384]              [1, 32, 384]              (148,608)                 148,608                   False
│    │    │    │    └─BertIntermediate (intermediate)                  --                        [1, 32, 384]              [1, 32, 1536]             --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 384]              [1, 32, 1536]             (591,360)                 591,360                   False
│    │    │    │    │    └─GELUActivation (intermediate_act_fn)        --                        [1, 32, 1536]             [1, 32, 1536]             --                        --                        --
│    │    │    │    └─BertOutput (output)                              --                        [1, 32, 1536]             [1, 32, 384]              --                        --                        False
│    │    │    │    │    └─Linear (dense)                              --                        [1, 32, 1536]             [1, 32, 384]              (590,208)                 590,208                   False
│    │    │    │    │    └─Dropout (dropout)                           --                        [1, 32, 384]              [1, 32, 384]              --                        --                        --
│    │    │    │    │    └─LayerNorm (LayerNorm)                       --                        [1, 32, 384]              [1, 32, 384]              (768)                     768                       False
│    └─BertPooler (pooler)                                             --                        [1, 32, 384]              [1, 384]                  --                        --                        False
│    │    └─Linear (dense)                                             --                        [1, 384]                  [1, 384]                  (147,840)                 147,840                   False
│    │    └─Tanh (activation)                                          --                        [1, 384]                  [1, 384]                  --                        --                        --
├─Sequential (proj)                                                    --                        [1, 384]                  [1, 256]                  --                        --                        True
│    └─Linear (0)                                                      --                        [1, 384]                  [1, 256]                  98,304                    98,304                    True
│    └─LayerNorm (1)                                                   --                        [1, 256]                  [1, 256]                  512                       512                       True
============================================================================================================================================================================================================================
Total params: 22,812,032
Trainable params: 98,816
Non-trainable params: 22,713,216
Total mult-adds (M): 22.81
============================================================================================================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 6.89
Params size (MB): 91.25
Estimated Total Size (MB): 98.14
============================================================================================================================================================================================================================
```