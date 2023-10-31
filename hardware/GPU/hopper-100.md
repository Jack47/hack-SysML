
Pytorch 下的用法：
```
import transformer_engine.pytorch as te
import transformer_engine.common import recipe

fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(inp)

loss = out.sum
loss.backward()
```

已经集成在多个框架里了：
