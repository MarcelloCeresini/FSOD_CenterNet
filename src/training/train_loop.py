import torch as T

from losses import heatmap_loss_batched, reg_loss_batched

def train_one_epoch(model,
                    training_loader,
                    optimizer,
                    # epoch_index, 
                    # tb_writer,
                    ):
    
    running_loss = 0.

    for i, (sample, transformed_landmarks, _) in enumerate(training_loader):
        loss = 0
        optimizer.zero_grad()

        n_detections = [len(l) for l in transformed_landmarks]

        input_image, labels = sample

        gt_reg, gt_heat_base, _ = labels        
        pred_reg, pred_heat_base, _ = model(input_image)

        loss += T.mean(heatmap_loss_batched(pred_heat_base,
                                              gt_heat_base,
                                              n_detections),
                       dim=0)
        
        loss += T.mean(reg_loss_batched(pred_reg,
                                        gt_reg,
                                        n_detections),
                       dim=0)
        
        loss.backward()
        optimizer.step()

        # which one?
        running_loss += loss.item()
        running_loss += loss

        # # TODO: do we need this?
        # # Gather data and report
        # if i % 1000 == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return running_loss