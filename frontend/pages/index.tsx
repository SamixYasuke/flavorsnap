/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from "react";
import Image from "next/image";


  );
}

export const getStaticProps: GetStaticProps = async ({ locale }) => ({
  props: {
    ...(await serverSideTranslations(locale ?? "en", ["common"])),
  },
});