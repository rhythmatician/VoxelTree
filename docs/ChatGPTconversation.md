



I’m wondering if our model has good potential for grokking

It might — but the key question is what you mean by “grokking.” In ML circles that word has a very specific meaning now, and it’s not the same as “the model works well.”

Let’s anchor the term first.

What “Grokking” Means in ML
The term comes from the paper Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.

In that context, grokking means:

The model first memorizes the training data.

Validation performance remains poor for a long time.

After many additional training steps, the model suddenly discovers the underlying rule and generalizes perfectly.

So the training curve typically looks like:

step →
train accuracy:  ███████████████████
val accuracy:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████
                                  ↑
                             sudden jump
This tends to happen when:

the dataset is small

the rule is simple but compositional

the model is overparameterized

weight decay or similar regularization is present

Does VoxelTree Have the Ingredients for Grokking?
Based on the design you’ve described for VoxelTree, some parts look promising.

1. Structured rules exist in terrain generation ✔
Minecraft terrain is not random.

It follows deterministic procedural rules driven by noise fields, biome logic, and cave carvers implemented in Minecraft worldgen.

Examples:

caves carve according to noise thresholds

rivers follow gradient minima

biomes impose block distributions

surfaces track heightmaps

Those are latent rules.

Your model could theoretically discover them.

2. Your inputs expose the latent structure ✔
You’ve already built several conditioning channels:

x_parent (8³ voxel context)

x_height

x_biome

x_lod

x_cave

x_rriver

That is extremely important.

Without those signals, the model would have to infer everything from voxels alone — much harder.

Instead, you’re basically saying:

“Here are the governing fields — now learn the mapping to blocks.”

That dramatically increases the chance the model learns rules instead of memorizing chunks.

3. The dataset is huge ❌ (this actually reduces grokking)
Classic grokking happens on small datasets.

You’re training on:

millions of voxel patches

across many seeds

across biomes

That usually causes models to learn gradually, not with a sudden phase transition.

So you probably won’t see classic grokking curves.

But you could still get something close to rule learning.

4. The architecture matters
If your model is:

convolutional

local receptive field

translation invariant

…it’s already biased toward learning procedural rules.

That’s good.

But grokking-style phase transitions tend to happen more with:

transformers

MLPs

large parameter counts

strong weight decay

The More Important Question
For LODiffusion, the goal isn’t really grokking.

The real goal is:

Learning the mapping from coarse terrain + environmental signals → plausible finer terrain.

Which is closer to conditional super-resolution than algorithm discovery.

In other words:

coarse terrain
+ biome
+ heightmap
+ cave mask
+ river mask
------------------
→ higher resolution terrain
That is more like:

image super-resolution

conditional generative modeling

structured upsampling

Where Grokking Could Appear
If it happens, you’d see something like this during training:

Early training

model memorizes local patterns

caves inconsistent

rivers broken

Mid training

terrain smooth

caves still noisy

Late training

suddenly caves align with masks

rivers follow valleys

biome boundaries respected

That would be the rule discovery moment.

One Interesting Experiment You Could Try
To intentionally encourage grokking-like behavior:

Train on a single Minecraft world seed.

Then evaluate on different seeds.

If the model generalizes well, it means it learned:

terrain rules

not terrain memorization

That’s a very strong signal.

My Honest Take
Your project actually has a surprisingly good chance of learning real structure because:

Minecraft worldgen is procedural and rule-based

your inputs expose those latent rules

voxel CNNs naturally learn local structure

So the model may end up learning things like:

“rivers carve valleys”

“stone depth correlates with height”

“air clusters in caves”

which is basically grokking the procedural generator.

If you want, I can also show you something interesting:

why Minecraft terrain might be one of the best possible datasets for grokking-like behavior in spatial models.

There’s a subtle reason related to noise fields and compositional structure that makes it unusually learnable.


Hit me with it

There’s a subtle property of Minecraft terrain generation that makes it unusually friendly for models to learn the underlying rules instead of just memorizing samples.

The short version:

Minecraft terrain is built from a small set of smooth, composable mathematical fields.

That structure is exactly the kind of thing neural networks tend to learn efficiently.

Let’s unpack why.

1. Minecraft terrain is basically a stack of continuous functions
Modern terrain generation in Minecraft is mostly built from combinations of noise functions.

Typical components include:

continentalness

erosion

peaks/valleys

temperature

humidity

cave noise

All of those are continuous scalar fields defined over space:

f(x, y, z) → value
The world is then generated by thresholding or blending those fields.

Example:

if density(x,y,z) > 0:
    place stone
else:
    air
or

block = biome_surface_rule(height, biome)
So the terrain is basically:

terrain(x,y,z) = F(noise1, noise2, noise3, ...)
That’s a smooth function with simple nonlinearities.

2. Neural networks are extremely good at approximating smooth functions
Neural nets approximate functions like:

f(x) = W2 * relu(W1 * x)
Stack enough layers and you get a universal approximator.

Your voxel CNN is essentially learning:

F(context, biome, heightmap, cave_mask) → blocks
If the underlying generator is composed of smooth noise fields and thresholds, that mapping is very learnable.

This is dramatically easier than something like:

photorealistic images

human language

arbitrary game logic

3. Minecraft terrain is locally deterministic
Another huge factor:

Local patches strongly determine their surroundings.

Example:

If a model sees this in an 8×8×8 context:

stone
stone
stone
stone
stone
It can infer:

underground

not near surface

probably not a river

cave probability depends on noise

Likewise if it sees:

grass
dirt
stone
It knows:

near surface

biome rules apply

caves unlikely above

That strong local signal dramatically reduces ambiguity.

4. Procedural noise has hidden periodic structure
Noise functions like Perlin and Simplex (used heavily in terrain generation) are compositions of periodic gradient fields.

Meaning there is hidden structure like:

sin-like oscillations
smooth gradients
fractal combinations
Those patterns repeat everywhere in the world.

That’s perfect training data because the model sees the same rules repeated millions of times.

Contrast that with natural images where every scene is unique.

5. Minecraft terrain is fractal (self-similar across scale)
Another huge advantage for your LOD hierarchy:

Minecraft terrain is roughly scale-invariant.

Large mountains look like scaled-up small hills.

Cave systems have similar structure across scales.

This means the mapping:

8³ → 16³
is very natural.

You’re essentially teaching the model a fractal refinement rule.

That’s exactly what CNNs are good at.

6. Your conditioning channels expose the latent fields
This is the really big one.

You’re giving the model:

biome

heightmap

cave mask

river mask

parent voxels

Those are almost the same signals the vanilla generator uses.

So the model doesn’t have to reverse-engineer the world generator from voxels alone.

Instead it learns something closer to:

block = f(parent_voxel, biome, height, cave_noise)
Which is extremely structured.

7. The model might learn better rules than vanilla
Here’s a fun possibility.

The vanilla generator uses hard thresholds like:

if noise > 0:
    stone
else:
    air
That produces artifacts:

floating islands

weird cave walls

jagged terrain transitions

A neural model might instead learn soft probabilistic rules:

P(stone) = sigmoid(...)
Which could produce more natural transitions.

8. Why this matters for LODiffusion
For your LOD system:

8³ parent voxels
+ biome
+ height
+ cave mask
+ river mask
-------------------
→ 16³ refined terrain
The model is learning the refinement operator of the terrain generator.

Almost like:

terrain_(n+1) = refine(terrain_n)
That’s extremely close to how:

fractal terrain algorithms

wavelet super-resolution

diffusion models

operate.

The surprising conclusion
Minecraft terrain may actually be one of the most learnable large-scale 3D datasets that exist.

It has:

massive data availability

strong local structure

smooth underlying functions

repeated patterns

fractal scaling

All of which are exactly what neural networks thrive on.

The really exciting possibility
If this works well, your model might not just reproduce terrain.

It might implicitly learn something close to the world generator itself.

Which means your LOD pipeline could end up behaving like a neural procedural generator.

And that’s the moment where the project stops being just a mod and starts looking like research.

If you want, I can also explain a counterintuitive risk in your architecture that could cause the model to learn too well and accidentally break world coherence at large distances.


Sure explain tha

The risk isn’t that the model fails to learn the terrain rules.

The risk is that it learns them too locally.

When that happens, the terrain can look perfectly plausible in small patches but become globally inconsistent across larger regions. For something like your LOD system, that can create visible artifacts when distant terrain eventually refines.

Let’s walk through why.

1. Your model only sees an 8³ parent context
Right now the refinement step is roughly:

8×8×8 parent voxels
→ predict
16×16×16 child voxels
That means the model’s decision horizon is roughly:

~8 blocks of spatial context
Anything larger than that must be inferred indirectly from:

biome

heightmap

cave mask

river mask

If those signals don’t fully encode the larger terrain structure, the model may invent plausible but inconsistent details.

Example problem:

patch A → generates a cave entrance
patch B → generates solid rock
Both look fine locally.

But globally the cave suddenly dead-ends at a chunk boundary.

2. Procedural terrain uses large spatial correlations
Minecraft worldgen often uses noise fields with wavelengths like:

32 blocks
64 blocks
128 blocks
So features like:

mountain ridges

valleys

cave networks

rivers

extend far beyond an 8-block context window.

The vanilla generator maintains consistency because those fields are evaluated globally.

Your model doesn’t see the full field.

It only sees samples of its consequences.

3. This creates a “locally correct, globally wrong” failure mode
A neural model may learn rules like:

caves often appear near caves
stone continues downward
rivers are low elevation
Those are good local rules.

But they don’t guarantee global continuity.

Example artifact:

LOD 3 terrain:

██████████████
████░░░███████
████░░░███████

LOD 0 refinement:

████░░░███████
░░░░░░░░░░░░░░
████░░░███████
The model decided the cave expands sideways, but the adjacent patch predicted rock.

Both predictions are individually plausible.

Together they’re impossible.

4. LOD systems amplify this problem
In progressive terrain refinement, inconsistencies can cascade.

Your pipeline roughly looks like:

LOD3 → LOD2 → LOD1 → LOD0
If an early level introduces a small structural error:

river slightly misplaced
cave slightly misaligned
ridge slightly shifted
then deeper refinements will commit to that mistake.

You may get:

rivers climbing uphill

caves clipping mountains

cliffs appearing mid-valley

5. CNNs are especially prone to this
Convolutional models are strongly local.

They excel at:

texture

local geometry

repeating patterns

They struggle with:

global constraints

long-distance structure

topology

This is why image super-resolution models sometimes create:

misaligned buildings

broken text

warped edges

Even though local detail looks fantastic.

6. Your conditioning channels partially fix this
The good news is that your design already mitigates this risk.

You include:

x_height
x_biome
x_cave
x_rriver
Those channels encode large-scale fields.

That gives the model anchors for global structure.

For example:

river mask = valley centerline
heightmap = global terrain slope
cave mask = large cave tunnels
That dramatically reduces the chance of incoherent terrain.

7. But there’s still one subtle failure mode
The model might treat conditioning inputs as soft suggestions, not hard constraints.

Example:

river mask = 1
model decides: “eh, maybe stone here”
Now the river disappears in that patch.

Or caves get blocked.

8. The safest solution (architecturally)
The most robust terrain ML systems do one of these:

Option A — Hard constraints
Use masks to force certain voxels.

Example:

if cave_mask == 1:
    voxel = air
Model predicts only the surrounding terrain.

Option B — Predict residuals
Instead of predicting terrain from scratch:

child = upsample(parent) + neural_residual
That keeps large structures stable.

Option C — Larger receptive field
Increase context window:

16³ parent → 32³ output
or add dilated convolutions.

My honest assessment of your design
You actually avoided most of the catastrophic cases because you already include:

biome

heightmap

cave mask

river mask

That’s unusually well designed for a terrain model.

Many generative terrain experiments fail because they only train on voxels.

You’re training on the fields that generated them.

That’s a huge advantage.

The most interesting emergent behavior to watch for
If the model learns correctly, you may see it start doing something fascinating:

It might smooth out the noise artifacts of vanilla terrain.

Meaning distant terrain generated by your model could actually look:

more coherent

more natural

less noisy

than terrain generated by the vanilla generator.

If that happens, you accidentally invented a neural terrain generator.

Which would be a pretty wild outcome for a Minecraft mod.

If you want, I can also show you a very weird property of Minecraft caves that might make them the hardest part of the model to learn — even harder than mountains or rivers.


Alright

Caves are tricky for a reason that doesn’t show up immediately when you look at voxel patches.

They’re not just geometry — they’re topology.

Mountains and rivers mostly behave like surfaces.
Caves behave like 3-D networks.

That difference matters a lot for a neural model.

1. Surface terrain is basically a height function
Most overworld terrain can be approximated as:

y = f(x, z)
Meaning every horizontal position has one dominant surface height.

So the model just learns patterns like:

grass
dirt
stone
stone
stone
That structure is simple and consistent.

Your x_height channel makes this even easier because it tells the model where the surface should be.

2. Caves are fundamentally different
Caves are volumes carved out of solid rock.

Instead of a surface, you have structures like:

rock
rock
air
air
rock
air
rock
And those air pockets connect into tunnels.

So the model must learn something closer to:

3-D connectivity
rather than a surface boundary.

3. Minecraft caves come from thresholded 3-D noise
Modern cave generation in Minecraft largely uses density functions like:

density(x,y,z) = noise(x,y,z)
Then the generator applies a threshold:

if density < T:
    air
else:
    stone
This produces percolation-style structures.

Meaning tunnels form when random fields cross certain thresholds.

The result is something like:

████░░██░░██
██░░░░░░░███
██░░██░░░███
████░░░░████
Those patterns are topologically sensitive.

4. Small prediction errors can break caves
If your model predicts just a few voxels wrong, it can change the cave topology.

Example:

True cave:

rock rock rock
rock air  rock
rock air  rock
rock air  rock
Predicted cave:

rock rock rock
rock air  rock
rock rock rock
rock air  rock
One voxel error sealed the tunnel.

Now the cave is two disconnected pockets.

Locally everything still looks reasonable.

Globally the cave system is broken.

5. This is a known problem in 3-D generative models
Voxel models often struggle with topology preservation.

They easily learn:

shapes

textures

density patterns

But they struggle with:

connectivity

loops

tunnels

branching networks

Which caves happen to be.

6. Your x_cave mask helps a lot
You already added a channel:

x_cave [1,1,8,8,8]
That’s extremely valuable.

It gives the model an external signal about where caves should exist.

Instead of discovering caves purely from voxels, the model can learn:

if cave_mask == 1:
    high probability of air
This stabilizes topology significantly.

7. But cave shape still has to be learned
Even with the mask, the model must decide:

tunnel thickness

branching

wall smoothness

floor vs ceiling

Those details depend on 3-D context.

If the receptive field is small, tunnels may look like:

jagged voxel blobs
instead of smooth passages.

8. The really weird property of Minecraft caves
Cave networks are close to something called percolation thresholds in physics.

Near that threshold:

tunnels connect suddenly

connectivity changes rapidly

tiny perturbations change topology

Meaning the generator is operating in a chaotic regime.

Small voxel changes can dramatically change the cave network.

This is fundamentally harder to learn than terrain surfaces.

9. The good news
Your architecture has several advantages:

x_cave mask provides topology hints

CNNs capture local 3-D structure well

voxel resolution is relatively low (16³)

So the model doesn't need perfect topology — it just needs plausible caves.

10. The real success metric
For LOD terrain generation, caves don’t need to match vanilla exactly.

They just need to satisfy:

continuous tunnels
reasonable thickness
consistent direction
Players won’t notice if the exact cave path differs from vanilla.

But they will notice if tunnels constantly:

dead-end

pinch shut

randomly open

One trick that could help a lot
If cave topology becomes unstable, a very simple fix is:

if cave_mask == 1:
    bias air probability strongly
This forces tunnels to stay open even if the model makes small mistakes.

Essentially you treat caves as semi-hard constraints.

Big picture
The model probably learns:

Surface terrain first

Biome block types next

Rivers and valleys

Caves last

Because caves require understanding 3-D connectivity, which is the hardest structure in the data.

If you’re curious, there’s also a very surprising reason rivers might actually be easier for the model to learn than mountains, even though they look more complex.


I’m starting to think full terrain generation and vanilla parity WITH performance gains is too ambitious if not impossible. If other people are ever gonna use this mod, they’re gonna want regular vanilla terrain generation, so our model should really only generate the Voxy terrain. Yes, this guarantees that performance will be worse than unmodded MC, but it also guarantees terrain generation parity…. We’d need a very, very accurate model before the terrain generation errors would become negligible as far as the player base is concerned. At least, that’s my take. What do you think? Performance is very important, critical even, but I think terrain parity is as well, at least at full playable resolution 

I think your take is probably right.

Not because the core idea is bad, but because you are now separating two very different goals that were previously bundled together:

playable-terrain replacement

distant-LOD hallucination / preview generation

Those are not equally hard.

Replacing vanilla terrain generation at full playable resolution, while also preserving vanilla parity closely enough that players don’t notice, is a much harsher requirement than generating plausible Voxy-style distant terrain.

My honest assessment
If the requirement is:

“At full playable resolution, users should effectively get normal vanilla terrain.”

then a learned model is the wrong component to put in charge of the final authoritative terrain, at least for a first public version.

Why:

vanilla parity is an extremely unforgiving target

tiny local errors become visible, navigational, and gameplay-relevant

players do not judge parity statistically; they judge it by moments of breakage

A model can be 99.5% “good” and still feel broken if it causes:

cave openings to move

rivers to shift

cliff lines to change

structures or surface composition to differ

weird transitions when real terrain arrives

For distant terrain, players tolerate approximation.
For terrain they can walk on, mine, build on, and compare against seed expectations, tolerance collapses.

The key asymmetry
For LOD preview terrain, the user asks:

“Does this look plausible from far away, and does it stream fast?”

For full-resolution terrain, the user asks:

“Is this the same world?”

That second question is brutal.

Even if the model produces terrain that is good, coherent, and even arguably prettier, it still fails if the player expected vanilla seed-faithful terrain.

That is why I think your revised framing is strategically stronger.

The safer product split
You’re basically converging on this architecture:

Vanilla remains authoritative for playable terrain
near player

simulation-relevant space

anything that will become interactable soon

anything that must preserve exact seed parity

Model generates only distant Voxy-style terrain
non-authoritative

visual proxy

maybe progressively improved

discarded/reconciled when vanilla terrain becomes available

That gives you:

exact parity where players care most

freedom to approximate where players mostly care about immersion and horizon continuity

That is a much more defensible design.

The tradeoff
You already named it correctly:

this probably cannot beat unmodded Minecraft on raw total generation cost

I agree. If vanilla terrain still has to be generated eventually, then the model is adding work, not replacing it.

So the performance argument has to shift.

Instead of:

“This makes Minecraft generate less terrain overall.”

the claim becomes:

“This makes far-distance rendering and exploration feel better, while controlling when and how expensive vanilla generation happens.”

That is still valuable. It is just a different value proposition.

Where performance can still matter
Even under this revised design, performance is still critical, but the target changes.

You no longer need the model to outperform vanilla at being the world generator.

You need it to outperform other approaches at providing convincing distant terrain cheaply enough to be worth having.

That means your bar becomes something like:

fast enough CPU inference for many distant patches

low memory footprint

stable visual continuity

graceful handoff when authoritative vanilla chunks load

no catastrophic stutter from the LOD system itself

That is a much narrower and more achievable optimization target.

Why I think this is the better public-facing strategy
If other people use the mod, many of them will care about one or more of these:

seed compatibility

multiplayer expectations

other worldgen mods

redstone / farms / structures eventually existing where expected

not feeling like the world “lies” to them

A model-generated distant horizon is acceptable as a visualization layer.

A model-generated playable world that subtly diverges from vanilla is much harder to sell.

Even users who like the idea in principle will get annoyed if the mod teaches them to expect one thing and then resolves into another.

The hardest failure mode
The worst case is not “terrain looks bad.”

It is:

“The distant terrain looked real enough that I trusted it, and then it changed when I got there.”

That is survivable for far LOD if the user understands it as approximate preview data.

It becomes unacceptable if the same system also claims to be the actual terrain generator at playable resolution.

That is why keeping vanilla authoritative near the player is such a strong boundary.

My recommendation
I would treat this as a product-line decision:

Version 1 target
Build the best possible neural distant terrain proxy system.

Success criteria:

horizon looks believable

transitions are smooth

vanilla terrain takes over cleanly near the player

performance overhead is acceptable

seed parity is preserved for authoritative terrain

Version 2 or research branch
Explore whether the model can replace some subset of vanilla generation safely, such as:

only non-authoritative intermediate LODs

only terrain classes with hard constraints

only biomes or dimensions where error tolerance is higher

only after strong parity benchmarks prove it

That lets you preserve ambition without making the whole mod depend on a much riskier premise.

The deeper strategic point
You do not actually need full neural terrain replacement to make this project interesting or useful.

If you can make distant terrain:

appear much sooner

look much more plausible

transition much more smoothly

avoid giant empty or wrong-looking horizons

that is already a meaningful win over many existing LOD systems.

And it is a win that does not require you to solve the nearly impossible problem of “vanilla-faithful neural worldgen at full playable resolution.”

My bottom line
I think:

terrain parity at playable resolution should beat model ambition

the model should probably generate only non-authoritative distant/Voxy terrain

vanilla should remain the ground truth wherever player interaction matters

performance still matters, but now as LOD-system efficiency, not total worldgen replacement

So yes — I think your revised position is the more realistic one.

It narrows the project, but in a good way. It moves it from “maybe impossible product promise” to “clear, defensible mod architecture.”

The one caveat is this: if the eventual handoff from model LOD to vanilla terrain is ugly or jarring, then even this safer design can still fail. So the transition logic may now matter more than the model itself.


We could also get some performance gains here by ignoring carvers, and even skipping underground subchunks 

Yes — and that is probably one of the strongest levers you have.

If you stop treating distant LOD terrain as “miniature full vanilla chunks” and instead treat it as selective visual terrain, you can claw back a lot of performance.

Why this helps
A big chunk of vanilla terrain cost and voxel complexity is tied up in things that barely matter at long range:

cave carving

underground air pockets

hidden subchunks below visible terrain

fine-grained subterranean topology

For distant rendering, most of that is wasted.

From far away, the player mainly needs:

skyline / silhouette

mountain and valley shape

visible cliff faces

exposed cave mouths when relevant

enough near-surface structure to avoid obvious lies

They do not need a faithful 3-D simulation of every underground void.

Ignoring carvers is a very plausible optimization
If the model’s Voxy terrain is only a visual proxy, then full cave parity at distance is probably not worth the cost.

You could simplify distant generation to something more like:

surface shell
+ a shallow visible subsurface band
+ optional exposed voids only where surface visibility implies them
instead of:

fully carved 3-D underground terrain
That is a huge reduction in complexity.

Skipping underground subchunks is even bigger
This is probably the more important optimization.

If a subchunk is fully buried and has no plausible visibility path from the outside, generating it at distant LOD is mostly pointless.

That means you can prioritize:

topmost terrain

side-exposed terrain on cliffs, ravines, coastlines

maybe a shallow depth budget below the surface

and skip:

deep sealed rock

enclosed cave networks

buried terrain with no line of sight

That matches your earlier design idea about selective subchunk rendering based on visibility.

This also reduces model burden
There’s another benefit here: it makes the ML problem easier.

A model asked to generate all distant terrain must learn:

surface structure

cave topology

underground continuity

deep rock composition

river cuts

transitions everywhere

A model asked to generate only what is visually relevant can focus on:

terrain envelope

exposed faces

plausible visible openings

coarse surface semantics

That is a much more tractable target.

The main risk
The danger is not technical purity — it’s visual betrayal.

If you skip carvers and underground subchunks too aggressively, players may notice things like:

a cliff face that should reveal a cave but doesn’t

terrain that looks “solid” from far away, then opens dramatically when approached

ravines appearing late

mountain sides changing more than expected during refinement

So I would not frame it as “ignore underground entirely.”

I’d frame it as:

generate only underground structure that has a meaningful chance of being visible before vanilla authoritative terrain takes over

That is a much safer rule.

A better mental model
Think of the distant terrain as a render proxy, not a world proxy.

A render proxy only needs to answer:

What can the player plausibly see?

What shape should the horizon have?

Which exposed surfaces matter?

That gives you permission to throw away a lot of hidden geometry.

A practical hierarchy
Something like this seems sensible:

Always generate
surface shell

topography

biome-colored surface materials

coastlines / rivers visible from above

cliff and ravine sidewalls if externally visible

Sometimes generate
cave mouths near exposed surfaces

large exposed overhangs

shallow visible voids near terrain boundaries

Usually skip
sealed underground subchunks

deep carver detail

fully enclosed cave systems

hidden subterranean refinement

That would preserve most of the visual value while cutting a lot of cost.

This probably aligns better with public expectations too
Most users will care much more about:

“Can I see believable far terrain?”

“Does the horizon pop in less?”

“Does it stutter less?”

than:

“Was there a distant buried cave pocket 90 blocks underground before I got there?”

So from a product perspective, this is a very good trade.

My honest take
I think this is one of the first simplifications you’ve mentioned that feels like a real architectural win, not just a compromise.

Why:

it preserves vanilla parity where it matters

it sharply reduces the scope of distant generation

it makes the model’s job easier

it creates real opportunities for performance savings

it fits the visibility-driven philosophy you already described

The strongest version of your mod may not be “AI generates all distant terrain.”

It may be:

AI generates only the visible, non-authoritative outer shell of distant terrain, while vanilla remains responsible for the actual world.

That is narrower, but much more defensible.

The next thing worth pinning down is a formal rule for when a subchunk counts as “visually relevant,” because that rule could become more important than the exact model architecture.


That’s true! I think maybe we can hook into Voxy for that. It already makes calls to RocksDB for latent terrain. We can just hook in and say, if it tries to get something that doesn’t exist yet, que it for generation, up to the LOD that Voxy requested 

Yes — that sounds much closer to a viable architecture.

And more importantly, it aligns the mod with actual demand instead of trying to predict everything ahead of time.

Why this is strong
If Voxy already treats latent terrain as something it fetches from RocksDB, then the cleanest integration point is exactly what you described:

Voxy requests terrain at some position and LOD.

Check cache / RocksDB.

If present, return it.

If missing, enqueue a generation job for that exact region and requested LOD.

Optionally return fallback data until the job completes.

That gives you a pull-driven generation system instead of a push-driven one.

That matters because it means:

you only generate what Voxy actually wants

you avoid wasting time on unseen terrain

LOD priority is naturally driven by renderer demand

your storage model stays aligned with Voxy’s own access pattern

Conceptually, this is much better than “generate the world”
It reframes the whole system from:

“We are replacing terrain generation”

to:

“We are serving missing latent terrain on demand”

That’s a much narrower promise, and a much more defensible one.

The main architectural advantage
This also gives you a natural contract:

Voxy owns:
deciding which regions / LODs are needed

requesting them

consuming returned latent terrain

Your system owns:
cache lookup

queueing missing requests

generating approximate terrain for the requested LOD

persisting results back to RocksDB

That separation is clean.

It means you do not need to invent your own global scheduler first.
You can piggyback on Voxy’s existing notion of what matters.

I’d strongly recommend one constraint
Do not treat “requested up to some LOD” as meaning “generate the whole chain immediately unless profiling proves it’s fine.”

That is the kind of thing that sounds efficient but can quietly explode workload.

Example danger:

Voxy requests LOD 4 for many distant regions

your system decides to generate LOD 4, 3, 2, and 1 immediately

suddenly the queue grows much faster than the renderer can benefit from it

Safer interpretation:

if Voxy requests LOD n, generate that requested LOD first

optionally schedule finer or coarser dependent levels only when there is a concrete reason

In other words, make the requested LOD the authoritative priority signal.

A good queue model
I’d think in terms of a deduplicated work queue keyed roughly by:

(dimension, region_x, region_z, lod)
with metadata like:

requested_at

priority

request_count

nearest_player_distance

movement-vector relevance

current status: missing / queued / generating / ready / failed

That way if Voxy asks for the same thing repeatedly, you don’t keep spawning jobs.

Likely states you’ll want
Something like this:

MISSING
→ QUEUED
→ GENERATING
→ READY
and maybe:

FAILED_TEMP
FAILED_HARD
STALE
STALE could matter if your generator logic or model version changes and cached terrain needs invalidation.

One subtle issue: fallback behavior
This is probably where the real integration pain will be.

When Voxy asks for latent terrain that does not exist yet, what happens right now?

You probably need one of these strategies:

Option A: return “not available yet”
Voxy renders nothing / lower fallback / fog / coarser proxy.

Option B: return a trivial placeholder
Something very cheap and obviously temporary.

Option C: block briefly
Probably dangerous for performance unless Voxy already tolerates it.

My guess is A or B is safer. Blocking the render path is exactly the kind of thing that can turn a neat architecture into stutter hell.

Another subtle issue: dependency direction
You’ll want to decide whether generating LOD n depends on LOD n+1.

If your model is true hierarchical refinement, then:

coarser LOD exists → refine to finer LOD
In that case, a request for LOD 2 may implicitly require LOD 3 first.

That is fine, but I would make that dependency explicit in the queue system rather than burying it in generation logic.

For example:

requested: LOD 2

missing prerequisite: LOD 3

enqueue LOD 3 at boosted priority

once ready, generate LOD 2

That gives you traceability instead of mystery stalls.

This also supports your earlier visibility-first philosophy
Because Voxy is already requesting terrain based on what it wants to render, hooking into those requests gives you a practical approximation of:

“generate only visually relevant terrain”

That is much better than trying to invent a visibility system from scratch on day one.

My main caution
This idea is strong if Voxy’s RocksDB requests are at the right abstraction level.

That is the assumption I would want to validate in code.

Specifically:

Is Voxy requesting data in clean chunk/region/LOD units?

Does a cache miss happen early enough for you to respond without blocking a hot path too badly?

Can you safely inject asynchronous generation without fighting Voxy’s threading model?

Does Voxy tolerate late-arriving terrain naturally, or will it expect immediate data?

Those are not theoretical details. They determine whether this is elegant or miserable.

So my honest take
As a design direction, this is one of your better ideas.

It gives you:

demand-driven generation

natural prioritization

no wasted terrain work

cleaner integration with Voxy’s storage model

a path to keeping vanilla authoritative elsewhere

The biggest unknown is not the ML model. It’s whether Voxy’s request path is friendly to asynchronous cache-miss generation.

That’s the part I would inspect next in the codebase.

A very practical next step would be to map the exact call chain:
Voxy requests latent terrain → cache lookup → miss handling → render fallback
because that will tell you whether this is a simple hook or a deep invasive patch.


Safer interpretation: • if Voxy requests LOD n, generate that requested LOD first • optionally schedule finer or coarser dependent levels only when there is a concrete reason

Well our architecture is iterative, so it would generate LODs up to N, but starting with the cheapest (coarsest) ones

Yeah — with an iterative coarse-to-fine pipeline, that changes the picture.

In that case, my earlier warning needs refinement:

If Voxy requests LOD N, and your generator must build coarser prerequisites first, then generating “up to N” is not scope creep — it’s just how the pipeline works.

So the real rule should be:

If Voxy requests LOD N, generate the minimum prerequisite chain needed to satisfy N, starting from the cheapest/coarsest missing level and progressing upward until N is available.

That is different from “generate everything because maybe it’ll be useful.”

Concretely
Suppose lower-detail/coarser means bigger LOD number:

LOD 4 = very coarse, cheap

LOD 3

LOD 2

LOD 1

LOD 0 = finest / near-playable

If Voxy requests LOD 2, and your model works like:

LOD 4 → LOD 3 → LOD 2
then the right behavior is something like:

Check whether LOD 2 exists.

If not, check whether its parent prerequisite exists.

Enqueue only the missing chain.

Execute from coarsest missing level downward toward the requested level.

Example:

LOD 4 exists

LOD 3 missing

LOD 2 missing

Then do:

generate LOD 3
generate LOD 2
stop
Not:

generate LOD 3
generate LOD 2
generate LOD 1
generate LOD 0
unless something explicitly asked for those.

Why this distinction matters
The danger I was pointing at was this pattern:

“A request for one LOD becomes an excuse to eagerly generate a bunch of other LODs that were not needed yet.”

Your version avoids that if it stops at the requested target.

That’s a good architecture.

So your coarse-first idea is actually sensible
Because coarse levels are:

cheaper

faster to produce

usable sooner

natural prerequisites for refinement

That gives you a nice property:

fast first response at rough detail

progressively better data as time allows

no need to solve the most expensive level first

That’s exactly the kind of staged behavior you want in a render-driven system.

The key implementation rule
I’d define queue semantics around a target LOD, not just a generic “generate this region.”

Something like:

request(region, target_lod = N)
Then the scheduler resolves that into:

generate all missing prerequisite LODs needed to satisfy N
in coarse-to-fine order
That way the system knows when it is done.

One subtle edge case
If Voxy later requests a finer LOD for the same region, you do not want to restart from scratch.

Example:

earlier request satisfied LOD 3

later request wants LOD 1

Then the chain becomes:

LOD 3 exists
generate LOD 2
generate LOD 1
So each generated level becomes cached scaffolding for future refinement.

That fits your iterative design really well.

My cleaned-up position
So I’d revise my earlier statement to this:

If Voxy requests LOD N, generate the missing prerequisite chain up to N, starting from the cheapest coarsest missing level. Do not automatically continue beyond N unless a new request or explicit background policy justifies it.

That version matches what you’re describing much better.

And honestly, that sounds like a pretty solid architecture.


One question. Will it be faster at runtime to have separate models for each refinement step, or one unified model?

Usually, separate models will be faster at runtime only if they are substantially smaller and simpler per step. Otherwise, one unified model is usually the better default.

The real answer depends on what “unified” means in your design.

The two options
Option A: one unified model
Same network handles all refinement steps, with LOD step encoded as an input like x_lod.

Conceptually:

child = model(x_parent, x_biome, x_height, x_lod, ...)
Option B: separate models per refinement step
Different ONNX model for each transition, for example:

lod4_to_lod3.onnx

lod3_to_lod2.onnx

lod2_to_lod1.onnx

lod1_to_lod0.onnx

Runtime speed: what usually matters most
At inference time on CPU, the biggest factors are usually:

parameter count

activation sizes

convolution count

cache friendliness

model-loading / session-management overhead

how well ONNX Runtime optimizes the graph

Not “one file vs many files” by itself.

So the question is really:

Does a unified model force enough extra complexity that each call becomes materially heavier?

My expectation for your case
For VoxelTree / LODiffusion, I would expect this:

One unified model is probably better if:
the tensor shapes are identical at every refinement step

the same kinds of terrain logic apply at each step

you can represent the step with x_lod

the model architecture stays small

That gives you:

one ONNX graph

one runtime session

simpler Java integration

simpler caching and metadata

more shared learning across levels

Separate models are probably better if:
different refinement steps truly need different behavior

coarse levels can use much smaller networks

fine levels need extra channels or more capacity

some steps can skip expensive detail logic entirely

That can save runtime because you are not paying for a “general” model when a tiny specialized one would do.

The hidden cost of separate models
Separate models sound attractive, but they add real systems cost:

1. Multiple ONNX sessions
In DJL / ONNX Runtime, multiple models usually means:

more session setup complexity

more memory overhead

more asset management

more integration surface for bugs

If the mod is constantly bouncing across LOD transitions, that can get annoying fast.

2. Harder caching and versioning
You now need to manage:

compatibility between model versions

per-step metadata

possibly different normalization schemes

test vectors for each step

3. More training and tuning burden
Instead of one model and one pipeline, now you have several.
That means more chances for one step to become the weak link.

For your project, this matters a lot because the mod integration is already hard enough.

The hidden cost of one unified model
A single model can become bloated.

If you make one network handle all steps, it may learn a compromise architecture that is:

larger than needed for coarse refinement

slower than necessary at distant LODs

harder to optimize per stage

That matters because the distant/coarse requests are likely the most frequent and latency-sensitive.

If 80% of requests are coarse LOD, and your unified model carries fine-detail machinery it doesn’t need, you lose.

My practical recommendation
I would start with one unified model, unless profiling proves otherwise.

Why:

simpler ONNX export

simpler DJL integration

simpler request pipeline

easier to debug

easier to benchmark apples-to-apples

easier to maintain parity across steps

For a first viable system, that simplicity is worth a lot.

Then benchmark it.

If runtime profiling shows that coarse LOD inference is too expensive, the first split I would try is not full per-step specialization, but a two-tier split:

Better compromise:
coarse model for distant LOD refinement

fine model for near-player refinement

That often gives most of the benefit without exploding complexity.

What I would expect architecturally
Given your goals, I suspect the best long-term setup may be:

Unified model for distant Voxy terrain
Used for:

coarse-to-mid LODs

visual proxy generation

fast repeated demand-driven inference

Vanilla authoritative generation for playable terrain
Used for:

LOD 0 or near-authoritative handoff

simulation-relevant chunks

In that world, you may not need a separate finest-step model at all, because vanilla takes over before that matters.

That would strongly favor one unified neural model for the Voxy side.

Rule of thumb
If the refinement steps are mostly the same task at different scales, use one model.

If the steps are genuinely different tasks, use different models.

For example:

“turn coarse terrain into slightly less coarse terrain” at several levels: likely one model

“turn distant shell proxy into near-playable block-accurate terrain”: probably not the same task

My honest guess for runtime
Assuming static shapes and a small Conv3D-style network:

One unified model is likely to be faster overall from a systems perspective.

Separate models might only be faster in raw inference if you aggressively shrink the coarse-stage networks.

So unless you already know the coarse and fine stages need very different capacities, I’d default to one unified model with x_lod conditioning.

The main caveat
This assumes the unified model does not require dynamic control flow or branchy per-LOD logic that hurts ONNX/DJL friendliness. If the architecture stays plain and static, unified is the safer bet.

The right next move is honestly pretty mechanical: estimate FLOPs and parameter counts for a plausible unified model versus a tiny-per-stage alternative, then benchmark exported ONNX inference on CPU. That will answer the question more reliably than intuition.


What’s with all the “if”s? You know it’s not always the same tensor size 

You're right to call that out. If the tensor sizes actually change between refinement steps, then the earlier “if shapes are identical” assumption doesn’t apply.

Once tensor sizes differ, the decision becomes much clearer.

With different tensor sizes, separate models are usually better
Your refinement steps look something like:

LOD4 → LOD3
8³ → 16³

LOD3 → LOD2
16³ → 32³

LOD2 → LOD1
32³ → 64³
(or whatever your exact hierarchy is).

A unified ONNX model cannot easily handle changing spatial sizes unless you do one of these:

Export with dynamic shapes

Always run the model at the largest size

Pad smaller inputs to the largest size

For your constraints, all three are bad.

Dynamic shapes
Possible, but:

harder for ONNX Runtime to optimize

worse CPU performance in many cases

more brittle integration with DJL

Always run the largest size
Example:

always run 64³ model
even when refining 8³ → 16³
That wastes huge amounts of compute.

Padding
Same problem — you're paying for empty voxels.

Separate models solve this cleanly
Each refinement step can be exported with fixed shapes, which ONNX and DJL like.

Example:

lod4_to_lod3.onnx
input  : [1, C, 8, 8, 8]
output : [1, N_blocks, 16, 16, 16]

lod3_to_lod2.onnx
input  : [1, C, 16, 16, 16]
output : [1, N_blocks, 32, 32, 32]
Benefits:

static shapes

optimal convolution kernels

better CPU cache behavior

simpler graph optimization

This aligns well with your <100 ms CPU inference constraint.

Runtime overhead of multiple models is small
The earlier concern about multiple sessions is mostly theoretical.

In practice:

you load each model once at mod startup

keep the sessions alive

inference calls are cheap relative to the convolution work

The real cost is the network itself, not the session switch.

It also matches your refinement architecture
Your system is already:

coarse → refine → refine → refine
So the model boundary matching the refinement boundary is natural.

Each step becomes a specialized operator:

Refine8
Refine16
Refine32
That’s conceptually cleaner too.

Another practical advantage
Different steps probably need different capacity.

Example:

Step	Difficulty	Needed model size
8³ → 16³	simple terrain envelope	small
16³ → 32³	surface detail	medium
32³ → 64³	caves / fine blocks	larger
If you force a unified model, everything must be sized for the hardest step.

That slows down the cheap steps unnecessarily.

For your architecture
Given:

iterative refinement

changing spatial sizes

ONNX export requirement

CPU inference target

DJL runtime

The most practical architecture is:

refine_8_to_16.onnx
refine_16_to_32.onnx
refine_32_to_64.onnx
Each:

fixed tensor shapes

optimized per step

loaded once

invoked by the scheduler when needed

One extra trick worth considering
Even with separate models, share the same channel schema:

x_parent
x_biome
x_height
x_lod
x_cave
x_rriver
That keeps your training pipeline and data loaders simple.

Only the spatial resolution changes.

So with the constraint you pointed out — different tensor sizes per refinement step — the answer is pretty straightforward:

Separate models will almost certainly be faster and easier to integrate.




