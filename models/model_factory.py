from models.context_encoder import ContextGenerator, ContextDiscriminator


def get_context_generator(bottleneck_dim, img_size: int = 64, out_size: int = 64, **kwargs):
    return ContextGenerator(bottleneck_dim=bottleneck_dim, img_size=img_size, out_size=out_size)


def get_context_discriminator(input_size: int = 64, **kwargs):
    return ContextDiscriminator(input_size=input_size)


MODEL_GETTERS = {
    "context_generator": get_context_generator,
    "context_discriminator": get_context_discriminator,
}
