use polars::prelude::arity::{
    broadcast_try_binary_elementwise, try_binary_elementwise, try_ternary_elementwise,
    try_unary_elementwise,
};
use polars::prelude::*;
use polars_arrow::array::Array;

#[inline]
pub fn try_unary_elementwise_values_with_dtype<'a, T, V, F, K, E>(
    ca: &'a ChunkedArray<T>,
    dtype: DataType,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: FnMut(T::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIterDtype<K>,
{
    if ca.null_count() == ca.len() {
        let arr = V::Array::full_null(ca.len(), dtype.to_arrow(CompatLevel::newest()));
        return Ok(ChunkedArray::with_chunk(ca.name().clone(), arr));
    }

    let iter = ca.downcast_iter().map(|arr| {
        let validity = arr.validity().cloned();
        let arr: V::Array = arr
            .values_iter()
            .map(&mut op)
            .try_collect_arr_with_dtype(dtype.to_arrow(CompatLevel::newest()))?;
        Ok(arr.with_validity_typed(validity))
    });
    ChunkedArray::try_from_chunk_iter(ca.name().clone(), iter)
}

#[inline]
#[allow(dead_code)]
pub fn try_binary_elementwise_values<T, U, V, F, K, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
{
    if lhs.len() == rhs.len() && (lhs.null_count() == lhs.len() || rhs.null_count() == rhs.len()) {
        let arr = V::Array::full_null(
            lhs.len(),
            V::get_static_dtype().to_arrow(CompatLevel::newest()),
        );

        return Ok(ChunkedArray::with_chunk(lhs.name().clone(), arr));
    }

    try_binary_elementwise(lhs, rhs, |a, b| match (a, b) {
        (Some(a), Some(b)) => Ok(Some(op(a, b)?)),
        _ => Ok(None),
    })
}

#[inline]
pub fn broadcast_try_binary_elementwise_values<T, U, V, F, K, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
{
    if lhs.len() == rhs.len() && (lhs.null_count() == lhs.len() || rhs.null_count() == rhs.len()) {
        let arr = V::Array::full_null(
            lhs.len(),
            V::get_static_dtype().to_arrow(CompatLevel::newest()),
        );

        return Ok(ChunkedArray::with_chunk(lhs.name().clone(), arr));
    }

    broadcast_try_binary_elementwise(lhs, rhs, |a, b| match (a, b) {
        (Some(a), Some(b)) => Ok(Some(op(a, b)?)),
        _ => Ok(None),
    })
}

#[inline]
pub fn broadcast_try_ternary_elementwise<T, U, G, V, F, K, E>(
    ca1: &ChunkedArray<T>,
    ca2: &ChunkedArray<U>,
    ca3: &ChunkedArray<G>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    G: PolarsDataType,
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
    ChunkedArray<U>: ChunkExpandAtIndex<U>,
    ChunkedArray<G>: ChunkExpandAtIndex<G>,
    V: PolarsDataType,
    F: for<'a> FnMut(
        Option<T::Physical<'a>>,
        Option<U::Physical<'a>>,
        Option<G::Physical<'a>>,
    ) -> Result<Option<K>, E>,
    V::Array: ArrayFromIter<Option<K>>,
{
    match (ca1.len(), ca2.len(), ca3.len()) {
        (1, 1, _) => {
            let a = unsafe { ca1.get_unchecked(0) };
            let b = unsafe { ca2.get_unchecked(0) };
            try_unary_elementwise(ca3, |c| op(a.clone(), b.clone(), c))
                .map(|ca| ca.with_name(ca1.name().clone()))
        }
        (1, _, 1) => {
            let a = unsafe { ca1.get_unchecked(0) };
            let c = unsafe { ca3.get_unchecked(0) };
            try_unary_elementwise(ca2, |b| op(a.clone(), b, c.clone()))
                .map(|ca| ca.with_name(ca1.name().clone()))
        }
        (_, 1, 1) => {
            let b = unsafe { ca2.get_unchecked(0) };
            let c = unsafe { ca3.get_unchecked(0) };
            try_unary_elementwise(ca1, |a| op(a, b.clone(), c.clone()))
        }
        (1, _, _) => {
            let ca1 = ca1.new_from_index(0, ca2.len());
            try_ternary_elementwise(&ca1, ca2, ca3, op)
        }
        (_, 1, _) => {
            let ca2 = ca2.new_from_index(0, ca1.len());
            try_ternary_elementwise(ca1, &ca2, ca3, op)
        }
        (_, _, 1) => {
            let ca3 = ca3.new_from_index(0, ca1.len());
            try_ternary_elementwise(ca1, ca2, &ca3, op)
        }
        _ => try_ternary_elementwise(ca1, ca2, ca3, op),
    }
}

#[inline]
pub fn broadcast_try_ternary_elementwise_values<T, U, G, V, F, K, E>(
    ca1: &ChunkedArray<T>,
    ca2: &ChunkedArray<U>,
    ca3: &ChunkedArray<G>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    G: PolarsDataType,
    ChunkedArray<T>: ChunkExpandAtIndex<T>,
    ChunkedArray<U>: ChunkExpandAtIndex<U>,
    ChunkedArray<G>: ChunkExpandAtIndex<G>,
    V: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>, G::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
{
    broadcast_try_ternary_elementwise(ca1, ca2, ca3, |a, b, c| match (a, b, c) {
        (Some(a), Some(b), Some(c)) => Ok(Some(op(a, b, c)?)),
        _ => Ok(None),
    })
}
