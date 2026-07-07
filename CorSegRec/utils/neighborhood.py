def neighbors_of(point, shape):
    z,y,x = point
    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    out = []
    for dz,dy,dx in offsets:
        nz,ny,nx = z+dz,y+dy,x+dx
        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
            out.append((nz,ny,nx))
    return out
