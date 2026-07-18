# Widgets

Widgets are the core building blocks of the demo product.

## Creating widgets

Use `WidgetFactory` to construct widgets.

```python
from demo import WidgetFactory

factory = WidgetFactory()
widget = factory.create(name="alpha")
```

Additional options are described in [configuration](./config.md).

## Lifecycle

Call `Widget.delete()` to release resources when a widget is no longer needed.
