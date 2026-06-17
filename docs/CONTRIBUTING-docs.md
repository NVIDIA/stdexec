# Contributing to stdexec's documentation

This guide is for anyone adding or editing API reference documentation,
user-guide content, or developer-guide content in stdexec. The goal is
that two contributors documenting different CPOs in parallel produce
documentation that looks like it was written by one person.

If you're documenting **one new CPO** end-to-end and need the short
version, jump straight to [the checklist](#checklist-adding-a-new-cpo).

## Where the docs live

Documenting a CPO touches up to three files:

| File                                                | What lives here                                                                                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `include/stdexec/__detail/<file>.hpp`               | Doxygen comments on the CPO type (`foo_t`) and its `inline constexpr foo_t foo{}` variable.                  |
| `docs/source/reference/index.rst`                   | A short orienting paragraph, a `_ref-<name>` label, and `breathe` directives to pull the Doxygen XML in.     |
| `docs/source/user/index.rst` *(consumer-facing)*    | An approachable, example-driven section keyed by a `_UserGuide_<name>` label. Skip if the CPO is internal.   |
| `docs/source/developer/index.rst` *(extender-facing)* | Only when the CPO is relevant to extending stdexec (custom algorithms, domains, schedulers, etc.).         |

The build pipeline is `CMake → Doxygen → Sphinx (with Breathe)`. Doxygen
parses C++ headers into XML; Breathe pulls those XML entries into
Sphinx via directives in the `.rst` files. To rebuild locally:

```sh
cmake -B build/docs -S . -DSTDEXEC_BUILD_DOCS=ON \
  -DSTDEXEC_BUILD_EXAMPLES=OFF -DSTDEXEC_BUILD_TESTS=OFF
cmake --build build/docs --target docs
# Output: build/docs/docs/index.html
```

## The CPO doc anatomy

Every public CPO has the same anatomy. Use [`__then.hpp`](../include/stdexec/__detail/__then.hpp)
and the rendered [`then` reference entry](./source/reference/index.rst) as
the canonical template. Each section below is a heading you should
include — in this order — unless the section truly does not apply.

### 1. `@brief` on the type

One sentence. Describes what the adaptor *does* in the abstract, not
*how*. Lead with "A pipeable sender adaptor that …" for adaptors,
"A sender factory that …" for factories, "A sender consumer that …" for
consumers.

```cpp
//! @brief A pipeable sender adaptor that transforms a predecessor sender's
//!        value completion by invoking a callable on the values it produces.
struct foo_t { /* … */ };
```

### 2. Detailed description (prose)

Two to four paragraphs of approachable prose immediately after `@brief`.
Cover *what it does* and *why you'd reach for it*, not internals. Include
both call syntaxes (direct and pipe) in a small `@code{.cpp}` block. End
with a pointer to the normative spec: `See [exec.foo] in the C++26
working draft for the normative specification.`

Use the exact working-draft stable name in brackets (e.g. `[exec.then]`,
`[exec.sync.wait.var]`). A Doxygen input filter (`eelis_link_filter.pl`)
automatically turns any `[exec.*]` reference into a hyperlink to the matching
section on <https://eel.is/c++draft> in the generated docs, so just write the
bare stable name — do not add the URL by hand.

### 3. Inline section headings

Use **bold inline headings** for sub-sections — not `@par`, not `###`.

```cpp
//! **Completion signatures.**
//!
//! Prose here, including code blocks, can span multiple paragraphs…
```

> ⚠️ **Why not `@par`?** Doxygen's `@par Title` only spans *one paragraph*,
> so any multi-paragraph section (which most are, because they include
> code blocks) leaks past the heading. Worse, the prose ends up
> rendered *before* the heading.
>
> ⚠️ **Why not markdown `###`?** Breathe 4.35 crashes on `docSect2TypeSub`
> nodes inside member docs.

Standard sub-section headings, in order:

1. `**Completion signatures.**` — Two code blocks (input sender → output
   sender), then any text explaining multiple-value-completion cases,
   `void`-returning callables, etc.
2. `**Exception behavior.**` — What happens if the callable throws or
   the predecessor errors out.
3. `**Cancellation.**` — Stop-token interaction and stopped-completion
   forwarding.
4. `**Example.**` — One complete, compilable example with `#include
   <stdexec/execution.hpp>`, a `main()`, and an `assert(…)` showing the
   expected result.

Omit a section only when there is genuinely nothing meaningful to say
(rare). Don't include a section as a stub.

### 4. `@see` cross-refs to related CPOs

End the type-level comment with `@see` lines pointing to closely related
CPOs. Use a single em-dash to introduce each one-line description:

```cpp
//! @see stdexec::upon_error  — adapt the error channel
//! @see stdexec::upon_stopped — adapt the stopped channel
//! @see stdexec::let_value   — adapt the value channel with a sender-returning function
```

Doxygen autolinks fully-qualified names that are also documented.
If the cross-referenced CPO isn't documented yet, leave the `@see`
in — it'll start resolving as soon as that CPO is documented.

### 5. Per-overload comments

Each `operator()` overload gets its own short comment block with
`@brief`, `@tparam`, `@param`, `@returns`, and any `@pre`:

```cpp
//! @brief Construct a sender that adapts @c __sndr by invoking @c __fun
//!        with each value-completion argument pack it produces.
//!
//! @tparam _Sender A type satisfying the @c stdexec::sender concept.
//! @tparam _Fun    A decayed, move-constructible callable type
//!                 (satisfying the internal <tt>__movable_value</tt> concept).
//!
//! @param __sndr   The predecessor sender ...
//! @param __fun    The function (or callable) to invoke ...
//!
//! @returns A sender that, when connected to a receiver and started, ...
//!
//! @pre @c __fun must be invocable with every value-completion argument
//!      pack of @c __sndr ...
template <sender _Sender, __movable_value _Fun>
constexpr auto operator()(_Sender&& __sndr, _Fun __fun) const
  -> __well_formed_sender auto;
```

Document **every overload**, including the pipeable (unary) form.

### 6. `@brief` on the inline variable

Keep this short — the substantive documentation lives on the type.

```cpp
//! @brief The customization point object for the @c foo sender adaptor.
//!
//! @c foo is an instance of @ref foo_t. See @ref foo_t for the full
//! description, the completion-signature transformation rules, exception
//! and cancellation behavior, and a usage example.
//!
//! @hideinitializer
inline constexpr foo_t foo{};
```

`@hideinitializer` suppresses Doxygen from dumping `{}` into the rendered
docs.

## Tag conventions

| Tag                | Use it for                                              | Notes                                                  |
| ------------------ | ------------------------------------------------------- | ------------------------------------------------------ |
| `@brief`           | One-sentence summary on every documented entity         | Required.                                              |
| `@tparam`          | Each template parameter                                 | Use the underscore-prefixed name (`_Sender`, `_Fun`).  |
| `@param`           | Each function parameter                                 | Use the underscore-prefixed name (`__sndr`, `__fun`).  |
| `@returns`         | The return value                                        | Prefer `@returns` over `@return` for symmetry.         |
| `@pre`             | Preconditions the caller must satisfy                   | Use for ill-formed-program conditions, e.g. invocability. |
| `@see`             | Cross-references to related CPOs                        | Fully qualify (`stdexec::let_value`).                  |
| `@code{.cpp}`      | C++ example blocks                                      | Always include the language tag for syntax highlighting. |
| `@c`               | Inline `<code>`-styled identifiers in prose             | `@c f`, `@c sndr`. Prefer this to backticks.           |
| `<tt>…</tt>`       | Inline code containing characters `@c` chokes on        | E.g. `<tt>then(sndr, f)</tt>` (has space and parens).  |
| `@hideinitializer` | Suppress `{}` from the rendered `inline constexpr ...`  | Use on every CPO inline-variable doc.                  |

Tags **not to use**: `@par` (multi-paragraph bug), `### markdown headings`
(breathe crash), `@param[in]` / `@param[out]` (we're const-correct via the
type, not via parameter direction tags).

## Documenting a function template (non-CPO)

A few public symbols are *function templates*, not customization-point
objects (e.g. `stdexec::get_completion_signatures<Sndr>()`). They have
no underlying struct, no `operator()`, and may be overloaded.

For these, neither `doxygenstruct` nor `doxygenvariable` applies, and
`doxygenfunction:: stdexec::name(args)` is finicky:

- For overloaded templates, breathe must disambiguate by signature.
  Specifying `doxygenfunction:: stdexec::name()` is ambiguous when
  multiple overloads have the same `argsstring`, and breathe fails with
  *"Unable to resolve function ... with arguments ()"*.
- For consteval templates whose entire signature is in template
  parameters, there's no string-level disambiguator.

The simplest workaround: render the *file's* documentation block with
`doxygenfile`:

```rst
.. doxygenfile:: __get_completion_signatures.hpp
   :sections: briefdescription detaileddescription
```

This pulls in the comments from the file and renders them inline; you
lose the per-overload separation but get the prose you wrote. Pair it
with explicit prose in the surrounding RST that names the two flavors
(`<Sndr>()` and `<Sndr, Env>()`) so readers know what they're looking
at.

## Documenting a C++20 concept

Concepts get the same anatomy as CPOs (brief, prose, requirements,
`@see`), with two differences:

- The concept itself is the target of the doxygen comment — there's no
  associated "type" page like there is for a CPO. Put all the
  documentation directly on the concept template.
- In the `.rst`, use `.. doxygenconcept::` (not `doxygenstruct`):

  ```rst
  .. _ref-concept-foo:

  ``foo``
  ^^^^^^^

  .. doxygenconcept:: stdexec::foo
  ```

When a concept depends on a tag type (e.g. `stdexec::sender_tag` for
`stdexec::sender`), document the tag type with `doxygenstruct` right
after the concept so both appear together.

## Cross-referencing

Between the three documentation files, links flow in both directions:

- **`.rst` → Doxygen entity**: use Sphinx C++ domain roles.
  `:cpp:member:`stdexec::foo`` for the inline variable,
  `:cpp:struct:`stdexec::foo_t`` for the type,
  `:cpp:func:`stdexec::foo_t::operator()`` for a specific overload.
- **Reference section ↔ User Guide**: define labels and reference them.
  In reference: `.. _ref-foo:` immediately before the section heading.
  In user guide: `.. _UserGuide_foo:` immediately before the section
  heading. Cross-reference with the **explicit-text form** of `:ref:`:

  ```rst
  see :ref:`sync_wait <ref-sync_wait>` for blah ...
  ```

  Avoid the bare form ``:ref:`ref-sync_wait` `` — even when the label
  is followed by a section heading, some such references fail with
  "A title or caption not found", apparently when other Sphinx-domain
  entities (e.g. `doxygenstruct`-rendered C++ symbols) get indexed
  against the same label. The explicit-text form bypasses the title
  lookup entirely and always works.
- **Doxygen → Doxygen**: `@see stdexec::other_cpo` autolinks when both
  are documented. Use `@ref foo_t` for an explicit ref to a struct/class.

## Reference section pattern

For each CPO, add a sub-heading and these directives to
`docs/source/reference/index.rst`:

```rst
.. _ref-foo:

``foo`` — one-line summary of what foo does
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One short paragraph orienting the reader, with a forward-link to the
User's Guide: see :ref:`UserGuide_foo` for an approachable introduction
with worked examples; the full reference follows.

.. doxygenstruct:: stdexec::foo_t
   :members:

.. doxygenvariable:: stdexec::foo
```

The `:members:` option on `doxygenstruct` is essential — without it,
Breathe renders only the brief and omits the `operator()` overloads.

Group adaptors by which completion channel they primarily affect (value
/ error / stopped), and within each group, list factory → adaptor →
consumer roles.

## User Guide section pattern

```rst
.. _UserGuide_foo:

``foo`` — one-line user-facing tagline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:cpp:member:`stdexec::foo` is the asynchronous counterpart to "…". You
give it … and you get back ….

The simplest possible example:

.. code-block:: cpp

    auto sndr = stdexec::just(21)
              | stdexec::foo([](int x) { return x * 2; });
    auto [v] = stdexec::sync_wait(std::move(sndr)).value();
    // v == 42
```

The structure to follow:

1. One paragraph: what it is and what you'd reach for it for.
2. Simplest possible example.
3. The two call syntaxes (direct vs. pipe).
4. A more realistic example: chained transformations, mixed types, etc.
5. **What happens on error?** (one paragraph)
6. **What happens on cancellation?** (one paragraph)
7. **When *not* to use it** — explicitly point readers to the alternative
   they probably want (e.g. `let_value` instead of `then` for
   sender-returning functions).

Bold-headed mini-sections are fine in RST; use the **`**Bold.**`**
convention so the visual style matches the Doxygen-rendered sections.

## Checklist: adding a new CPO

For each new CPO `foo`, in order:

1. ☐ Add Doxygen on the `foo_t` type per [the anatomy](#the-cpo-doc-anatomy).
2. ☐ Add Doxygen on each `operator()` overload.
3. ☐ Add a short `@brief` + `@ref foo_t` + `@hideinitializer` on the
   `inline constexpr foo_t foo{}` variable.
4. ☐ Add the `.. _ref-foo:` block to `docs/source/reference/index.rst`.
5. ☐ Add the `.. _UserGuide_foo:` block to `docs/source/user/index.rst`
   (if the CPO is consumer-facing).
6. ☐ Add a developer-guide entry to `docs/source/developer/index.rst`
   only if the CPO is relevant to extending the framework.
7. ☐ Run `cmake --build build/docs --target docs` and confirm:
   - Build succeeds with no *new* warnings (3 pre-existing warnings are
     OK: 2 intersphinx-network, 1 `abi-breaks.md` toctree).
   - The new section renders with no empty `<dd><p></p>` blocks (sign
     of misused `@par`).
   - Cross-refs in both directions resolve to working links (no
     `??` or unresolved `:cpp:member:` warnings).

## Special case: CPOs whose underlying type is `__`-prefixed

A few CPOs are instances of types that themselves start with a double
underscore — e.g. `read_env` is `inline constexpr __read_env_t read_env{}`.
Because `EXCLUDE_SYMBOLS = *__*` hides the type, there is *no* `_t` page
for Breathe to surface; the only public symbol Sphinx can render is the
`inline constexpr` variable.

In this situation:

- **Omit the `doxygenstruct` directive** from the reference `.rst`. The
  underlying type isn't documented and shouldn't appear in the index.
- **Put the substantive prose on the `inline constexpr` variable**
  directly — `@brief`, the long description, *Completion signatures*,
  *Exception behavior*, *Cancellation*, *Example*, `@see`. Add
  `@hideinitializer` to suppress the `{}` from the rendered signature.
- **Skip per-overload `@tparam`/`@param`/`@returns` blocks** — there's no
  type-level page to attach overload documentation to. Describe the call
  form inline in the type-level prose (a `@code{.cpp}` block showing
  `read_env(q)` is usually enough).
- The rendered signature will leak the internal type name (e.g.
  `__read_env_t const stdexec::read_env`). This is a known cosmetic
  wart — accept it, or rename the type publicly if the CPO is important
  enough.

`read_env` is the canonical example of this pattern.

## Special case: CPOs that inherit `operator()` from a private base

Some CPO types share their `operator()` overloads by inheriting from a
common implementation base — e.g. `let_value_t` / `let_error_t` /
`let_stopped_t` all inherit `operator()` from `__let::__let_t<…>`. Because
`EXCLUDE_SYMBOLS = *__*` hides the base, Breathe's `:members:` option will
only surface the (boring) defaulted default-constructor on the derived
type — not the overloads that are the actual API.

In this situation:

- **Drop `:members:`** from the `doxygenstruct` directive in the reference
  `.rst`. Listing the defaulted constructor is just noise.
- **Show the call signatures inline in the type-level prose** using a
  `@code{.cpp}` block before the *Completion signatures* section, e.g.:

  ```cpp
  //! The signature of the operator overloads (inherited from a detail base) is:
  //!
  //! @code{.cpp}
  //! template <sender Sender, movable-value Fun>
  //!   auto operator()(Sender&& sndr, Fun fun) const -> sender auto;   // direct
  //!
  //! template <class Fun>
  //!   auto operator()(Fun fun) const;                                 // closure
  //! @endcode
  ```
- **Skip the per-overload `@tparam` / `@param` / `@returns` blocks** —
  there's nowhere for them to attach. Put any precondition info in the
  type-level prose instead.

You can usually spot this case at a glance: the derived type's struct body
contains only a `using` and a defaulted constructor.

## Sphinx anchor normalization

Sphinx slugifies labels for HTML IDs: `_ref-let_value:` in the RST becomes
`id="ref-let-value"` in the rendered HTML. Two practical consequences:

- When grepping the rendered HTML, search for the dashed form
  (`ref-let-value`, `userguide-then`), not the underscored form from the
  `.rst` label.
- Cross-references in RST (`:ref:`UserGuide_let_value``) still use the
  original underscored label — Sphinx does the slugification.

## Toolchain wrinkles you might trip over

### Pragma macros confuse the doxygen parser

`STDEXEC_PRAGMA_IGNORE_GNU(...)`, `STDEXEC_PRAGMA_PUSH()`,
`STDEXEC_PRAGMA_POP()`, etc. expand (via `_Pragma`) to GCC/Clang
pragma operators. When one of these macros sits at file scope ahead of
a heavily-commented type definition, doxygen's preprocessor can fail to
recognize the subsequent struct as a class, leaving you with a hpp-level
XML file that has no `innerclass` entries — and breathe will then warn
that it cannot find `stdexec::your_t`.

The fix is in the Doxyfile (`docs/Doxyfile.in`): predefine each pragma
macro to an empty token sequence so doxygen's preprocessor strips it
out entirely. The current set is:

```
"STDEXEC_PRAGMA_PUSH()= "
"STDEXEC_PRAGMA_POP()= "
"STDEXEC_PRAGMA_IGNORE_GNU(X)= "
"STDEXEC_PRAGMA_IGNORE_EDG(X)= "
"STDEXEC_PRAGMA_IGNORE_MSVC(X)= "
```

If you add a new pragma macro elsewhere in the codebase, add it to
`PREDEFINED` in the Doxyfile too — otherwise the first contributor who
documents a type below it will see breathe warnings and wonder why.

### Variadic macros need `(...)` in PREDEFINED, not `(X)`

When predefining a function-like macro, the parameter list must *match the
arity of the actual macro*. `STDEXEC_ATTRIBUTE` is defined in
`__config.hpp` as `STDEXEC_ATTRIBUTE(...)` (variadic) and is called both
as `STDEXEC_ATTRIBUTE(always_inline)` and as
`STDEXEC_ATTRIBUTE(host, device)`. If the Doxyfile predefines it as
`STDEXEC_ATTRIBUTE(X)=` (single-arg), the two-arg call mis-substitutes
and leaks the second argument into the surrounding signature — you end
up with rendered prototypes like

```
device constexpr auto stdexec::just_t::operator()(_Ts &&... __ts) const
```

which Sphinx's C++ domain then refuses to parse, surfacing as

```
reference/index.rst:NN: WARNING: Error when parsing function declaration.
```

— annoyingly, the source location Sphinx reports is the `.rst` line of
the `doxygenstruct` directive, *not* the offending macro. So if you see
that warning, suspect a PREDEFINED macro-arity mismatch, not the `.rst`.

The fix is in the Doxyfile: predefine the macro with `(...)` so it
absorbs any number of arguments:

```
"STDEXEC_ATTRIBUTE(...)= "
```

If you introduce a new variadic macro in the codebase, mirror it in
`PREDEFINED` the same way.

## Pitfalls (the short list)

- **Don't use `@par`** — use `**Bold inline headings.**` instead.
- **Don't use markdown `###` inside member docs** — breathe crashes.
- **Don't forget `:members:` on `doxygenstruct`** — overloads vanish.
  *Exception:* when the operator overloads come from an excluded base,
  drop `:members:` and document the signatures in prose (see above).
- **Don't put the substantive doc on the `inline constexpr` variable** —
  `doxygenvariable` doesn't render the type's `operator()` overloads. Put
  the long description on the type and a brief pointer on the variable.
- **Don't document internals** — symbols matching `*__*` are already
  excluded by the Doxyfile (`EXCLUDE_SYMBOLS = *__*`). Don't fight this
  by giving an `_Impl`-suffixed type its own brief.
- **Beware `@ref` followed by punctuation.** Doxygen consumes the next
  token wholesale, so `@ref starts_on_t:` is parsed as the symbol name
  `starts_on_t:` (with the colon) and fails to resolve. Prefer an em-dash
  or comma with a leading space (`@ref starts_on_t — note: …`), or use
  the explicit-text form `@ref starts_on_t "starts_on_t"`.
- **No Unicode in `@code{.cpp}` blocks.** Pygments' C++ lexer can't
  tokenize `…`, `→`, em-dashes inside code, etc., and Sphinx warns
  about "Lexing literal_block ... resulted in an error at token …".
  Use ASCII (`...`, `->`, `--`) inside code blocks. Em-dashes etc. are
  fine in prose.
- **No `<tt>X<Y></tt>` for templated names.** Doxygen's HTML parser
  treats the inner `<Y>` as an HTML tag (and a single-capital `<S>`
  becomes a strikethrough), corrupting the rest of the comment block.
  Symptoms include "found `</tt>` tag while expecting `</s>`" warnings,
  doxygen losing track of subsequent class definitions in the file
  (resulting in `Cannot find class "stdexec::foo_t"` warnings from
  breathe), and concept/struct symbols losing their `stdexec::`
  namespace in the XML. Use **markdown backticks** for any inline code
  containing angle brackets: write `` `enable_sender<S>` `` instead of
  `<tt>enable_sender<S></tt>`. (Backticks are opaque to doxygen's HTML
  parser.)
- **No `/* ... */` inside `@code{.cpp}` blocks.** Doxygen's
  comment scanner gets confused by `*/` even inside what it should
  treat as opaque code, and reports "reached end of file while inside
  a 'code' block". Use `// ...` line comments inside example code.

## Marking a CPO as deprecated

When a CPO is deprecated (e.g. `transfer_when_all`), document it both
in the doxygen comment and in the `.rst`:

- **In the doxygen comment**, use `@deprecated` with a one-line
  recommendation pointing at the replacement. Note that Doxygen
  captures `@deprecated` as an `xrefsect` and renders it on a *global*
  "Deprecated List" page — **it does not appear inline on the struct's
  reference page**. So `@deprecated` alone is not enough for visibility.
- **In the `.rst`**, add a `.. admonition::` block right above the
  `doxygenstruct` directive so the warning renders inline:

  ```rst
  .. admonition:: Deprecated
     :class: warning

     This adaptor is not part of the C++26 working draft and is retained
     only for backwards compatibility. Write
     ``when_all(sndrs...) | continues_on(sch)`` instead.
  ```

  Don't use `.. deprecated::` — that directive treats the *first word*
  of its argument as the "since version" string, so
  `.. deprecated:: This adaptor is …` renders as "Deprecated since
  version This: adaptor is …", which is gibberish.
