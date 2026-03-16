package com.voxeltree.harvester.mixin;

import com.voxeltree.harvester.DataHarvesterClient;
import net.minecraft.client.gui.screens.TitleScreen;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Logs when the title screen initializes, confirming the mod is loaded
 * and the auto-connect system is active.
 */
@Mixin(TitleScreen.class)
public class TitleScreenMixin {

    @Inject(method = "init", at = @At("RETURN"))
    private void onTitleScreenInit(CallbackInfo ci) {
        DataHarvesterClient.LOGGER.info(
                "[DataHarvester] Title screen detected. Auto-connect is {}.",
                DataHarvesterClient.getConfig().autoConnect ? "ARMED" : "DISABLED"
        );
    }
}
