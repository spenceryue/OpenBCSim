#include "device_properties.h"

DeviceProperties &get_properties (int device)
{
  static DeviceProperties props;
  static int device_ = -1;
  if (device != device_)
  {
    checkCall (cudaGetDeviceProperties (&props, device));
    device_ = device;
  }
  return props;
}

#define DEVICE_PROPERTY(name, description)                                               \
  Class.def_property_readonly (#name,                                                    \
                               [](const DeviceProperties &self) {                        \
                                 return sanitize_member (self, &DeviceProperties::name); \
                               },                                                        \
                               description);

#define DEVICE_DICT_ENTRY(name, _) \
  Class.attr ("__dict__")[#name] = sanitize_member (self, &DeviceProperties::name);

void bind_DeviceProperties (py::module m)
{
  auto Class = py::class_<DeviceProperties> (m, "DeviceProperties", "Struct with CUDA device properties.",
                                             py::dynamic_attr ());
  Class.def (py::init ([](int device = 0) {
               auto copy = DeviceProperties (get_properties (device));
               return copy;
             }),
             "device"_a = 0);
  FORALL_DEVICE_PROPERTIES (DEVICE_PROPERTY)

  Class.def ("__repr__", [=](const DeviceProperties &self) {
    auto dict = py::cast (self).attr ("__dict__");
    if (py::len (dict) == 0)
    {
      FORALL_DEVICE_PROPERTIES (DEVICE_DICT_ENTRY)
    }
    return py::str (dict);
  });
}
