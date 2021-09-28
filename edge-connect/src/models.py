import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, BinarizationLoss, TVLoss
from .modules import GradientScaler
from .tools import get_gradient_ratios


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        binarization_loss = BinarizationLoss()
        perceptual_loss = PerceptualLoss()
        tv_loss = TVLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('binarization_loss', binarization_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('tv_loss', tv_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks, scaler=None):
        self.iteration += 1
        #confmap.cuda()


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        #outputs = outputs**2
        gen_loss = 0
        dis_loss = 0


        if self.config.OSGAN:
            # One Stage GANs
            input_real = torch.cat((images, edges), dim=1)
            input_fake = torch.cat((images, outputs), dim=1)
            fake_neg = scaler(input_fake)

            pred_real, real_feat = self.discriminator(input_real)        # in: (grayscale(1) + edge(1))
            pred_fake, fake_feat = self.discriminator(fake_neg)        # in: (grayscale(1) + edge(1))

            real_loss = self.adversarial_loss(pred_real, True, True)
            fake_loss = self.adversarial_loss(pred_fake, False, True) # reductionいる？

            #dis_loss += (real_loss + fake_loss) / 2
            dis_loss += real_loss + torch.mean(fake_loss)

            gen_gan_loss = self.adversarial_loss(pred_fake, True, True)
            gen_loss += torch.mean(gen_gan_loss)

            gamma = get_gradient_ratios(gen_gan_loss, fake_loss, pred_fake)[:,0,0,0]
            grad_d_factor = 1.0 / (1.0 - gamma)

            loss_pack_fake = fake_loss - gen_gan_loss
            scaled_loss_pack_fake = loss_pack_fake * grad_d_factor
            loss_pack = real_loss + torch.mean(scaled_loss_pack_fake)

            GradientScaler.factor = gamma
        
        else:
            # discriminator loss
            dis_input_real = torch.cat((images, edges), dim=1)
            dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
            dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
            dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake = torch.cat((images, outputs), dim=1)
            gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
            gen_loss += gen_gan_loss

            # generator feature matching loss
            gen_fm_loss = 0
            for i in range(len(dis_real_feat)):
                gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
            gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
            gen_loss += gen_fm_loss

            # generator binatization loss
            gen_bi_loss = self.binarization_loss(outputs)
            gen_loss += gen_bi_loss

            # generator total variation loss
            gen_tv_loss = self.tv_loss(outputs)
            gen_loss += gen_tv_loss

            # generator perceptual loss
            #edges_3ch = edges.expand(-1,3,-1,-1)
            #outputs_3ch = outputs.expand(-1,3,-1,-1)
            #gen_content_loss = self.perceptual_loss(outputs_3ch, edges_3ch)
            #gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            #gen_loss += gen_content_loss

            loss_pack = None


        #create logs
        if self.config.OSGAN:
            logs = [
                ("l_p", loss_pack.item()),
                ("l_d1", dis_loss.item()),
                ("l_g1", gen_loss.item()),
            ]
        else:
            logs = [
                ("l_d1", dis_loss.item()),
                ("l_g1", gen_gan_loss.item()),
                ("l_fm", gen_fm_loss.item()),
                ("l_bi", gen_bi_loss.item()),
                ("l_tv", gen_tv_loss.item())
            ]

        return outputs, gen_loss, dis_loss, logs, loss_pack

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None, loss_pack=None):
        if loss_pack is not None:
            loss_pack.backward()
            self.dis_optimizer.step()
            self.gen_optimizer.step()

        else:
            if dis_loss is not None:
                dis_loss.backward()
            self.dis_optimizer.step()

            if gen_loss is not None:
                gen_loss.backward()
            self.gen_optimizer.step()



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        tv_loss = TVLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('tv_loss', tv_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator adversarial loss
        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)

        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)# * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss
        
        # generator total variation loss
        #gen_tv_loss = self.tv_loss(outputs)
        #gen_loss += gen_tv_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
            #("l_tv", gen_tv_loss.item()),
        ]
            
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

