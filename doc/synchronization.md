# Synchronization

With Vulkan, GPU operations are executed asynchronically.

In principle, vulkpy automatically `wait()` depending `Job`
before reading or destructing, and users don't need to `wait()` explicitly.

In order to keep necessary resources during GPU execution,
the result `Array` holds them, too.

Just in case some arrays get circular reference and memory won't be released,
users might call `wait()` explicitly to clear reference of depending resources.
