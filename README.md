# Online Matching with Stochastic Rewards: Better Bounds and Robust Algorithms via Adversarial Reinforcement Learning

This paper gives an attempt to address these issues.
We focus on a concrete problem, named Online Matching with Stochastic Rewards} (OMSR), which generalizes the classic OBM model.
This model is more practical to the online advertising platform, where payments are given by advertisers only when their ads are clicked by a user.
The click-through-rate can be used as an estimation of the probability of those clicks.
Only when the user actually clicks the ad, which is dominated by a stochastic process, the advertiser should pay for it.
The randomness coming from the problem itself brings remarkable challenges in theoretical analysis.
While an optimal $1-1/e \approx 0.632$-competitive algorithm is known to OBM, OMSR remains open: an upper bound of 0.621 Meaning that OBSR is strictly harder than OBM.

Our paper studies OMSR and considers both sides of learning robust algorithms and hard instances.
We set up an adversarial reinforcement learning framework for OMSR.
The framework consists of an iterative process between an adversary agent _adv_ and an algorithm agent _alg_: it starts from a simple algorithm, named Balance; _adv_ learns the hardest instances which make the current _alg_ perform the worst; _alg_ then learns for a better performance over these hard instances \footnote{More precisely, for robustness, _alg_ should be trained over a mixture of hard instances and some randomly generated instances.
Compared to most previous work, our framework can not only learn a high-performance algorithm at the very last, but more importantly, learn theoretically provable hardness results for OMSR during the iterative processes as well.
We conclude our main results as follows:

- For hardness results, we prove that there is no algorithm for OMSR with a competitive ratio of more than $0.597$, inspired by a family of instances learned by _adv_.
- For designing algorithms, we empirically show that our algorithms learned by _alg_ performs better than the state-of-the-art (SOTA) algorithm, Balance.

We finally evaluate our framework on a well-studied problem, AdWords , for which an optimal $(1-1/e)$-competitive algorithm has been known.
Experimental results validate that our framework can converge to an optimal in the end.

## installation guide

## rust install

### Toolchain management with `rustup`

Rust is installed and managed by the [`rustup`](https://rust-lang.github.io/rustup/) tool. Rust has a 6-week [rapid release process](https://github.com/rust-lang/rfcs/blob/master/text/0507-release-channels.md) and supports a [great number of platforms](https://forge.rust-lang.org/release/platform-support.html), so there are many builds of Rust available at any time. `rustup` manages these builds in a consistent way on every platform that Rust supports, enabling installation of Rust from the beta and nightly release channels as well as support for additional cross-compilation targets.

If you've installed `rustup` in the past, you can update your installation by running `rustup update`.

For more information see the [`rustup` documentation](https://rust-lang.github.io/rustup/).

### Configuring the `PATH` environment variable

In the Rust development environment, all tools are installed to the `~/.cargo/bin` directory, and this is where you will find the Rust toolchain, including `rustc`, `cargo`, and `rustup`.

Accordingly, it is customary for Rust developers to include this directory in their [`PATH` environment variable](<https://en.wikipedia.org/wiki/PATH_(variable)>). During installation `rustup` will attempt to configure the `PATH`. Because of differences between platforms, command shells, and bugs in `rustup`, the modifications to `PATH` may not take effect until the console is restarted, or the user is logged out, or it may not succeed at all.

If, after installation, running `rustc --version` in the console fails, this is the most likely reason.

### Libtorch Manual Install

Get libtorch from the PyTorch website download section and extract the content of the zip file.

For Linux users, add the following to your .bashrc or equivalent, where /path/to/libtorch is the path to the directory that was created when unzipping the file.

```bash

export LIBTORCH=/path/to/libtorch

export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

```

The header files location can also be specified separately from the shared library via the following:

#### LIBTORCH_INCLUDE must contains `include` directory.

`export LIBTORCH_INCLUDE=/path/to/libtorch/`

#### LIBTORCH_LIB must contains `lib` directory.

`export LIBTORCH_LIB=/path/to/libtorch/`

For Windows users, assuming that `X:\path\to\libtorch` is the unzipped libtorch directory.

Navigate to Control Panel -> View advanced system settings -> Environment variables.

Create the LIBTORCH variable and set it to `X:\path\to\libtorch`.

Append X:\path\to\libtorch\lib to the Path variable.
