pub fn try_reduce<I, F, T, E>(mut iter: I, mut func: F) -> Result<Option<T>, E>
where
    I: Iterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let Some(mut acc) = iter.next() else {
        return Ok(None);
    };
    for item in iter {
        acc = func(acc, item)?;
    }
    Ok(Some(acc))
}
