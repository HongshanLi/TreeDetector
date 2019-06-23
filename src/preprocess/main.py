import csv
import os

# project root directory
root_dir= "../../"

def _create_img_ids(img_dir):
    '''create ids for each raw HD imgs'''
    img_ids = []
    for x in os.listdir(img_dir):
        img_ids = img_ids + os.listdir(
                os.path.join(img_dir, x))
    return img_ids

def preprocess_imgs():
    '''divide raw HD images (1250x1250) into subimgs of dim 250 x 250
    Only keep rgb channels
    
    '''
    raw_image_mask = os.path.join(
            root_dir, "raw_image_mask.csv"
            )
    with open(raw_image_mask, "r") as f:
        csv


    
    num = 1250 // 250
    for img_id in img_ids:
        # img sub-folder
        img_sfd = img_id.split('-')[0]

        # rgb
        img_rgb = img_id + "_RGB-Ir.tif"
        
        # mask
        img_mask = img_id + "_TREE.png"
        
        img_path = os.path.join(img_dir, 
                img_sfd, img_id, img_rgb)


        mask_path = os.path.join(img_dir,
                img_sfd, img_id, img_mask)

        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        # divid into subimgs and save 
        for i in range(num):
            for j in range(num):
                start_x, end_x = i*250, (i+1)*250
                start_y, end_y = j*250, (j+1)*250
                
                sub_img = img[start_x:end_x, start_y:end_y,0:3]

                sub_mask = mask[start_x:end_x, start_y:end_y,0:3]
                
                idx = '{}_{}'.format(i,j)
                file_name = img_id + '_' + idx + ".png"

                # save img
                Image.fromarray(sub_img).save(
                        os.path.join(out_dir, 'imgs/', file_name)
                        )

                # save mask
                Image.fromarray(sub_mask).save(
                        os.path.join(out_dir, 'masks/', file_name)
                        )
    print("Num of imgs processed:", len(img_ids))


def compute_mean_std(dataset):
    '''compute channelwise mean and std of the dataset
    Arg:
        dataset: pytorch Dataset where each data is a tensorized PIL img
    '''
    batch_size=5
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    start = time.time()

    # compute means for each channels
    mean = 0

    for i,(img, elv, mask) in enumerate(loader):
        mean = mean + torch.mean(img, dim=[0,2,3])
    
    mean = mean /(i+1)
    
    # compute std
    cum_var = 0
    _mean = mean.view(1, 3, 1, 1).repeat(batch_size, 1, 250,250) 
    for i,(img, elv, mask) in enumerate(loader):
        # repeat means across channels
        
        try:
            cum_var = cum_var + torch.mean((img - _mean)**2, dim=[0,2,3])
        except:
            print('current img:', img.shape)
            print('current mean', _mean.shape)
    var = cum_var / (i + 1)

    std = torch.sqrt(var)
    
    return mean, std

def main():


