package com.voxeltree.harvester.ingest;

import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.network.RegistryFriendlyByteBuf;
import net.minecraft.network.codec.StreamCodec;
import net.minecraft.network.protocol.common.custom.CustomPacketPayload;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.ChunkPos;

import java.util.ArrayList;
import java.util.List;

/**
 * S2C network payload carrying serialised chunk sections for Voxy ingestion.
 *
 * <p>Wire format is identical to VoxyWorldGen v2's {@code LODDataPayload}:
 * ChunkPos, minY, then a VarInt-length list of {@link SectionData} entries
 * (each containing section Y, serialised block-state palette, serialised
 * biome palette, and optional block/sky light layers).
 *
 * <p>By using the exact same serialisation that VoxyWorldGen v2 sends,
 * the client-side Voxy ingestion path is guaranteed to produce results
 * identical to normal gameplay — absolute parity.
 */
public record IngestPayload(ChunkPos pos, int minY, List<SectionData> sections)
        implements CustomPacketPayload {

    public static final Identifier ID = Identifier.parse("dataharvester:ingest_chunk");
    public static final Type<IngestPayload> TYPE = new Type<>(ID);
    public static final StreamCodec<RegistryFriendlyByteBuf, IngestPayload> CODEC =
            CustomPacketPayload.codec(IngestPayload::write, IngestPayload::new);

    /** Per-section data: serialised PalettedContainers + optional light. */
    public record SectionData(int y, byte[] states, byte[] biomes,
                              byte[] blockLight, byte[] skyLight) {

        public void write(RegistryFriendlyByteBuf buf) {
            buf.writeInt(y);
            buf.writeByteArray(states);
            buf.writeByteArray(biomes);
            buf.writeNullable(blockLight, (b, a) -> b.writeByteArray(a));
            buf.writeNullable(skyLight, (b, a) -> b.writeByteArray(a));
        }

        public static SectionData read(RegistryFriendlyByteBuf buf) {
            return new SectionData(
                    buf.readInt(),
                    buf.readByteArray(),
                    buf.readByteArray(),
                    buf.readNullable(b -> b.readByteArray()),
                    buf.readNullable(b -> b.readByteArray()));
        }
    }

    /** Deserialise from network. */
    public IngestPayload(RegistryFriendlyByteBuf buf) {
        this(buf.readChunkPos(), buf.readInt(),
                buf.readCollection(ArrayList::new,
                        b -> SectionData.read((RegistryFriendlyByteBuf) b)));
    }

    /** Serialise to network. */
    public void write(RegistryFriendlyByteBuf buf) {
        buf.writeChunkPos(pos);
        buf.writeInt(minY);
        buf.writeCollection(sections,
                (b, s) -> s.write((RegistryFriendlyByteBuf) b));
    }

    @Override
    public Type<? extends CustomPacketPayload> type() {
        return TYPE;
    }
}
