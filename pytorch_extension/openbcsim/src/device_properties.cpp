#include "device_properties.h"

DLL_PUBLIC DeviceProperties &get_properties (int device)
{
  static DeviceProperties props;
  static int device_ = -1;
  if (device != device_)
  {
    cudaDeviceProp tmp;
    checkCall (cudaGetDeviceProperties (&tmp, device));
    props = tmp;
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
  __dict__[#name] = sanitize_member (self, &DeviceProperties::name);

void bind_DeviceProperties (py::module m)
{
  auto Class = py::class_<DeviceProperties> (m, "DeviceProperties",
                                             "Struct with CUDA device properties.\n"
                                             "Call __repr__() to populate __dict__.",
                                             py::dynamic_attr ());
  Class.def (py::init ([](int device = 0) {
               return DeviceProperties (get_properties (device));
             }),
             "device"_a = 0);

  FORALL_DEVICE_PROPERTIES (DEVICE_PROPERTY)

  Class.def ("__repr__", [=](const DeviceProperties &self) {
    auto __dict__ = py::cast (self).attr ("__dict__");
    if (py::len (__dict__) == 0)
    {
      FORALL_DEVICE_PROPERTIES (DEVICE_DICT_ENTRY)
    }
    return py::str (__dict__);
  });

  Class.def ("__getitem__", [](const DeviceProperties &self, py::object key) {
    return py::cast (self).attr ("__dict__")[key];
  });
}
