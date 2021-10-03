## std::move
Eliminating spurious copies. 

## reinterpret_cast
```
reinterpret_cast<new_type>(expression)
```

Returns a value of type new\_type


```
const float4 *input4 = reinterpret_cast<const float4 *>(input);
```
